"""This file defines the model."""
import os.path
import logging
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .io_utils import pianoroll_to_image, vector_to_image
from .io_utils import image_grid, save_pianoroll
from .losses import get_adv_losses
from .utils import load_component, make_sure_path_exists
from ..base import BaseModel

LOGGER = logging.getLogger(__name__)

def get_scheduled_variable(start_value, end_value, start_step, end_step, current_step):
    """Return a scheduled decayed/growing variable."""
    if start_step > end_step:
        raise ValueError("`start_step` must be smaller than `end_step`.")
    if start_step == end_step:
        return torch.tensor(start_value)

    schedule_step = max(0, current_step - start_step)
    if schedule_step >= end_step - start_step:
        return torch.tensor(end_value)

    # Linear interpolation
    progress = schedule_step / (end_step - start_step)
    return start_value + progress * (end_value - start_value)

class museGAN(BaseModel):
    """Class that defines the model."""
    def __init__(self, params, name='Model'):
        super(museGAN, self).__init__()
        self.name = name

        # Build the model
        LOGGER.info("Building model.")
        if params.get('is_accompaniment'):
            self.gen = load_component(
                'generator', params['nets']['generator'], 'Generator')(
                    n_tracks=params['data_shape'][-1] - 1,
                    condition_track_idx=params['condition_track_idx'])
        else:
            self.gen = load_component(
                'generator', params['nets']['generator'], 'Generator')(
                    n_tracks=params['data_shape'][-1])
        self.dis = load_component(
            'discriminator', params['nets']['discriminator'],
            'Discriminator')(
                n_tracks=params['data_shape'][-1],
                beat_resolution=params['beat_resolution'])

        # Move models to the appropriate device
        self.gen.to(self.device)
        self.dis.to(self.device)

        # Save components to a list for showing statistics
        self.components = [self.gen, self.dis]

        # Initialize steps counter
        self.register_buffer('muse_global_step', torch.tensor(0, dtype=torch.int32))
        self.register_buffer('muse_gen_step', torch.tensor(0, dtype=torch.int32))

    def __call__(self, x=None, z=None, y=None, c=None, mode=None, params=None,
                 config=None):
        if mode == 'train':
            if x is None:
                raise TypeError("`x` must not be None for 'train' mode.")
            return self.get_train_nodes(x, z, y, c, params, config)
        elif mode == 'predict':
            if z is None:
                raise TypeError("`z` must not be None for 'predict' mode.")
            return self.get_predict_nodes(z, y, c, params, config)
        raise ValueError("Unrecognized mode received. Expect 'train' or "
                         "'predict' but get {}".format(mode))

    def get_train_nodes(self, x, z=None, y=None, c=None, params=None,
                        config=None):
        """Return a dictionary of objects for training."""
        LOGGER.info("Building training nodes.")

        nodes = {}

        # Move inputs to device
        if x is not None:
            x = x.to(self.device)
        if y is not None:
            y = y.to(self.device)
        if c is not None:
            c = c.to(self.device)

        # Set default latent distribution if not given
        if z is None:
            nodes['z'] = torch.randn(config['batch_size'],
                                     params['latent_dim'],
                                     device=self.device)
        else:
            nodes['z'] = z.to(self.device)

        # Get slope tensor (for straight-through estimators)
        self.register_buffer('slope', torch.tensor(1.0, device=self.device))
        nodes['slope'] = self.slope

        # --- Generator output ---------------------------------------------
        if params['use_binary_neurons']:
            if params.get('is_accompaniment'):
                nodes['fake_x'], nodes['fake_x_preactivated'] = self.gen(
                    nodes['z'], y, c, True, nodes['slope'])
            else:
                nodes['fake_x'], nodes['fake_x_preactivated'] = self.gen(
                    nodes['z'], y, True, nodes['slope'])
        else:
            if params.get('is_accompaniment'):
                nodes['fake_x'] = self.gen(nodes['z'], y, c, True)
            else:
                nodes['fake_x'] = self.gen(nodes['z'], y, True)

        # --- Slope annealing ----------------------------------------------
        if config['use_slope_annealing']:
            slope_schedule = config['slope_schedule']
            scheduled_slope = get_scheduled_variable(
                1.0, slope_schedule['end_value'], slope_schedule['start'],
                slope_schedule['end'], self.muse_global_step.item())
            self.slope.copy_(scheduled_slope)

        # --- Discriminator output -----------------------------------------
        nodes['dis_real'] = self.dis(x, y, True)
        nodes['dis_fake'] = self.dis(nodes['fake_x'], y, True)

        # ============================= Losses =============================
        LOGGER.info("Building losses.")
        # --- Adversarial losses -------------------------------------------
        nodes['gen_loss'], nodes['dis_loss'] = get_adv_losses(
            nodes['dis_real'], nodes['dis_fake'], config['gan_loss_type'])

        # --- Gradient penalties -------------------------------------------
        if config['use_gradient_penalties']:
            batch_size = x.size(0)
            eps_shape = [batch_size] + [1] * (len(x.shape) - 1)
            eps_x = torch.rand(eps_shape, device=self.device)
            inter_x = eps_x * x + (1.0 - eps_x) * nodes['fake_x']
            inter_x.requires_grad_(True)
            dis_x_inter_out = self.dis(inter_x, y, True)

            gradients = torch.autograd.grad(
                outputs=dis_x_inter_out,
                inputs=inter_x,
                grad_outputs=torch.ones_like(dis_x_inter_out),
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            slopes_x = torch.sqrt(1e-8 + torch.sum(
                gradients ** 2, dim=list(range(1, len(gradients.shape)))))
            gradient_penalty_x = torch.mean((slopes_x - 1.0) ** 2)
            nodes['dis_loss'] += 10.0 * gradient_penalty_x

        # Compute total loss (for logging and detecting NAN values only)
        nodes['loss'] = nodes['gen_loss'] + nodes['dis_loss']

        # ========================== Optimizers ==========================
        LOGGER.info("Building optimizers.")
        # --- Learning rate setup ------------------------------------------
        nodes['learning_rate'] = torch.tensor(
            config['initial_learning_rate'], device=self.device)

        if config['use_learning_rate_decay']:
            scheduled_learning_rate = get_scheduled_variable(
                config['initial_learning_rate'],
                config['learning_rate_schedule']['end_value'],
                config['learning_rate_schedule']['start'],
                config['learning_rate_schedule']['end'],
                self.muse_global_step.item())
            nodes['learning_rate'] = scheduled_learning_rate

        # --- Optimizers ---------------------------------------------------
        nodes['gen_optimizer'] = optim.Adam(
            self.gen.parameters(),
            lr=nodes['learning_rate'].item(),
            betas=(config['adam']['beta1'], config['adam']['beta2']))

        nodes['dis_optimizer'] = optim.Adam(
            self.dis.parameters(),
            lr=nodes['learning_rate'].item(),
            betas=(config['adam']['beta1'], config['adam']['beta2']))

        # --- Training functions --------------------------------------------
        def train_discriminator():
            nodes['dis_optimizer'].zero_grad()
            nodes['dis_loss'].backward(retain_graph=True)
            nodes['dis_optimizer'].step()
            self.muse_global_step += 1

        def train_generator():
            nodes['gen_optimizer'].zero_grad()
            nodes['gen_loss'].backward()
            nodes['gen_optimizer'].step()
            self.muse_gen_step += 1

        nodes['train_ops'] = {
            'dis': train_discriminator,
            'gen': train_generator
        }

        return nodes

    def get_predict_nodes(self, z=None, y=None, c=None, params=None,
                          config=None):
        """Return a dictionary of objects for prediction."""
        LOGGER.info("Building prediction nodes.")

        # Move inputs to device
        if z is not None:
            z = z.to(self.device)
        if y is not None:
            y = y.to(self.device)
        if c is not None:
            c = c.to(self.device)

        nodes = {'z': z}

        # Get slope tensor (for straight-through estimators)
        self.register_buffer('slope', torch.tensor(1.0, device=self.device))
        nodes['slope'] = self.slope

        # --- Generator output ---------------------------------------------
        with torch.no_grad():
            if params['use_binary_neurons']:
                if params.get('is_accompaniment'):
                    nodes['fake_x'], nodes['fake_x_preactivated'] = self.gen(
                        nodes['z'], y, c, False, nodes['slope'])
                else:
                    nodes['fake_x'], nodes['fake_x_preactivated'] = self.gen(
                        nodes['z'], y, False, nodes['slope'])
            else:
                if params.get('is_accompaniment'):
                    nodes['fake_x'] = self.gen(nodes['z'], y, c, False)
                else:
                    nodes['fake_x'] = self.gen(nodes['z'], y, False)

        # ============================ Save ops ============================
        def _get_filepath(folder_name, name, suffix, ext):
            """Return the filename."""
            if suffix:
                return os.path.join(
                    config['result_dir'], folder_name, name,
                    '{}_{}.{}'.format(name, str(suffix, 'utf8'), ext))
            return os.path.join(
                config['result_dir'], folder_name, name,
                '{}.{}'.format(name, ext))

        def _array_to_image(array, colormap=None):
            """Convert an array to an image array and return it."""
            if array.ndim == 2:
                return vector_to_image(array)
            return pianoroll_to_image(array, colormap)

        # --- Save array ops -----------------------------------------------
        if config['collect_save_arrays_op']:
            def _save_array(array, suffix, name):
                """Save the input array."""
                filepath = _get_filepath('arrays', name, suffix, 'npy')
                # Convert tensor to numpy array
                if torch.is_tensor(array):
                    array = array.detach().cpu().numpy()
                np.save(filepath, array.astype(np.float16))

            def save_arrays():
                arrays = {'fake_x': nodes['fake_x']}
                if params['use_binary_neurons']:
                    arrays['fake_x_preactivated'] = nodes['fake_x_preactivated']

                for key, value in arrays.items():
                    _save_array(value, config['suffix'], key)
                    make_sure_path_exists(
                        os.path.join(config['result_dir'], 'arrays', key))

            nodes['save_arrays_op'] = save_arrays

        # --- Save image ops -----------------------------------------------
        if config['collect_save_images_op']:
            def _save_image_grid(array, suffix, name):
                image = image_grid(array, config['image_grid'])
                filepath = _get_filepath('images', name, suffix, 'png')
                imageio.imwrite(filepath, image)

            def _save_images(array, suffix, name):
                """Save the input image."""
                # Convert tensor to numpy array
                if torch.is_tensor(array):
                    array = array.detach().cpu().numpy()

                if 'hard_thresholding' in name:
                    array = (array > 0).astype(np.float32)
                elif 'bernoulli_sampling' in name:
                    rand_num = np.random.uniform(size=array.shape)
                    array = (.5 * (array + 1.) > rand_num)
                    array = array.astype(np.float32)
                images = _array_to_image(array)
                _save_image_grid(images, suffix, name)

            def _save_colored_images(array, suffix, name):
                """Save the input image."""
                # Convert tensor to numpy array
                if torch.is_tensor(array):
                    array = array.detach().cpu().numpy()

                if 'hard_thresholding' in name:
                    array = (array > 0).astype(np.float32)
                elif 'bernoulli_sampling' in name:
                    rand_num = np.random.uniform(size=array.shape)
                    array = (.5 * (array + 1.) > rand_num)
                    array = array.astype(np.float32)
                images = _array_to_image(array, config['colormap'])
                _save_image_grid(images, suffix, name)

            def save_images():
                images = {'fake_x': .5 * (nodes['fake_x'] + 1.)}
                if params['use_binary_neurons']:
                    images['fake_x_preactivated'] = .5 * (
                        nodes['fake_x_preactivated'] + 1.)
                else:
                    images['fake_x_hard_thresholding'] = nodes['fake_x']
                    images['fake_x_bernoulli_sampling'] = nodes['fake_x']

                for key, value in images.items():
                    _save_images(value, config['suffix'], key)
                    _save_colored_images(value, config['suffix'], key + '_colored')
                    make_sure_path_exists(os.path.join(
                        config['result_dir'], 'images', key))
                    make_sure_path_exists(os.path.join(
                        config['result_dir'], 'images', key + '_colored'))

            nodes['save_images_op'] = save_images

        # --- Save pianoroll ops -------------------------------------------
        if config['collect_save_pianorolls_op']:
            def _save_pianoroll(array, suffix, name):
                filepath = _get_filepath('pianorolls', name, suffix, 'npz')
                # Convert tensor to numpy array
                if torch.is_tensor(array):
                    array = array.detach().cpu().numpy()

                if 'hard_thresholding' in name:
                    array = (array > 0)
                elif 'bernoulli_sampling' in name:
                    rand_num = np.random.uniform(size=array.shape)
                    array = (.5 * (array + 1.) > rand_num)
                save_pianoroll(
                    filepath, array, config['midi']['programs'],
                    list(map(bool, config['midi']['is_drums'])),
                    config['midi']['tempo'], params['beat_resolution'],
                    config['midi']['lowest_pitch'])

            def save_pianorolls():
                if params['use_binary_neurons']:
                    pianorolls = {'fake_x': nodes['fake_x'] > 0}
                else:
                    pianorolls = {
                        'fake_x_hard_thresholding': nodes['fake_x'],
                        'fake_x_bernoulli_sampling': nodes['fake_x']}

                for key, value in pianorolls.items():
                    _save_pianoroll(value, config['suffix'], key)
                    make_sure_path_exists(
                        os.path.join(config['result_dir'], 'pianorolls', key))

            nodes['save_pianorolls_op'] = save_pianorolls

        return nodes

    @torch.no_grad()
    def sample(self, batch_size: int):
        """Generate dummy samples for museGAN to satisfy BaseModel."""
        # This stub returns an empty tensor; sampling functionality can be added later
        return torch.tensor([], device=self.device)

    def training_step(self, batch, batch_idx):
        """Minimal training step stub."""
        # return zero loss to satisfy Lightning
        loss = torch.tensor(0.0, device=self.device)
        self.log("train/loss", loss)
        return loss

    def configure_optimizers(self):
        """Stub optimizer configuration."""
        return []

