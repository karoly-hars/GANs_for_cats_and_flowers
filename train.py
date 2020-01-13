import json
import argparse
import random
import os
import torch
from torch.utils.data import DataLoader
from utils import save_image_grid
from cat_dataset import Catfaces64Dataset
from flowers_dataset import Flowers64Dataset
from networks import Generator, DCGANDiscriminator, WGANDiscriminator, weights_init


def init_training(args):
    """Initialize networks, optimizers and the data pipeline."""
    # create dataset
    dataset = {
        'cats': Catfaces64Dataset.create_from_scratch,
        'flowers': Flowers64Dataset.create_from_scratch
    }[args.dataset](args.data_path)
    print('num of images:', len(dataset))

    # load config
    config_path = {
        'dcgan': 'dcgan_train_config.json',
        'wgan_gp': 'wgan_gp_train_config.json'
    }[args.gan_type]
    with open(config_path, 'r') as json_file:
        train_config = json.load(json_file)

    # define data loader
    data_loader = DataLoader(
        dataset, batch_size=train_config['batch_size'], shuffle=True, num_workers=train_config['num_workers']
    )

    # check CUDA availability and set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use GPU: {}'.format(str(device) != 'cpu'))

    # init networks
    generator = Generator()  # the generator is the same for both the DCGAN and the WGAN
    generator.apply(weights_init)
    generator = generator.to(device)

    discriminator = {
        'dcgan': DCGANDiscriminator,
        'wgan_gp': WGANDiscriminator
    }[args.gan_type]()
    discriminator.apply(weights_init)
    discriminator = discriminator.to(device)

    # Optimizers
    optimizers = {
        'gen': torch.optim.Adam(
            generator.parameters(), lr=train_config['learning_rate_g'], betas=(train_config['b1'], train_config['b2'])
        ),
        'disc': torch.optim.Adam(
            discriminator.parameters(), lr=train_config['learning_rate_d'], betas=(train_config['b1'], train_config['b2'])
        )
    }

    # make save dir, if needed
    os.makedirs(os.path.join(args.checkpoint_path, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(args.checkpoint_path, 'samples'), exist_ok=True)

    # load weights if the training is not starting from the beginning
    if train_config['start_epoch'] > 1:
        gen_path = os.path.join(
            args.checkpoint_path, 'weights', 'checkpoint_ep{}_gen.pt'.format(train_config['start_epoch']-1)
        )
        disc_path = os.path.join(
            args.checkpoint_path, 'weights', 'checkpoint_ep{}_disc.pt'.format(train_config['start_epoch']-1)
        )
        generator.load_state_dict(torch.load(gen_path, map_location=device))
        discriminator.load_state_dict(torch.load(disc_path, map_location=device))

    return device, data_loader, train_config, generator, discriminator, optimizers


def training_step_dcgan(batch, device, generator, discriminator, optimizers, train_config, loss_fn):
    """Run training step of the generator and the discriminator in a DCGAN architecture."""
    imgs = batch.to(device)

    # Sample noise as generator input
    z = torch.randn(imgs.size()[0], train_config['latent_dim'], 1, 1).to(device)

    valid = torch.ones(imgs.size(0)).to(device)
    fake = torch.zeros(imgs.size(0)).to(device)

    # -------------------
    # Train Discriminator
    # -------------------
    optimizers['disc'].zero_grad()
    # Sample real
    real_loss = loss_fn(discriminator(imgs), valid)
    # Sample fake
    gen_imgs = generator(z)  # generate fakes
    fake_loss = loss_fn(discriminator(gen_imgs.detach()), fake)
    # Backprop.
    d_loss = real_loss + fake_loss
    d_loss.backward()
    optimizers['disc'].step()

    # ---------------
    # Train generator
    # ---------------
    optimizers['gen'].zero_grad()
    g_loss = loss_fn(discriminator(gen_imgs), valid)
    # Backprop.
    g_loss.backward()
    optimizers['gen'].step()

    return g_loss, d_loss


def training_step_wgan_gp(batch_idx, batch, device, train_config, generator, discriminator, optimizers):
    """Run training step of the generator and the discriminator in a WGAN architecture with gradient penalty loss."""
    imgs = batch.to(device)

    # Sample noise as generator input
    z = torch.randn(imgs.size()[0], train_config['latent_dim'], 1, 1).to(device)

    one, neg_one = torch.tensor(1.0).to(device), torch.tensor(-1.0).to(device)
    u = torch.Tensor(imgs.size()[0], 1, 1, 1).uniform_(0, 1).to(device)
    grad_outputs = torch.ones(imgs.size()[0]).to(device)

    # -------------------
    # Train Discriminator
    # -------------------
    discriminator.zero_grad()
    # Sample real
    loss_real = discriminator(imgs)
    loss_real = loss_real.mean()
    loss_real.backward(neg_one)
    # Sample fake
    gen_imgs = generator(z).detach()
    loss_fake = discriminator(gen_imgs)
    loss_fake = loss_fake.mean()
    loss_fake.backward(one)

    # Gradient penalty
    interpolates = (u * imgs + (1 - u) * gen_imgs).to(device)
    interpolates.requires_grad_(True)
    grad = torch.autograd.grad(
        outputs=discriminator(interpolates),
        inputs=interpolates,
        grad_outputs=grad_outputs,
        retain_graph=True,
        create_graph=True,
        only_inputs=True
    )
    grad = grad[0]
    grad_penalty = train_config['grad_penalty_weight'] * ((grad.norm(2, 1).norm(2, 1).norm(2, 1) - 1) ** 2).mean()
    grad_penalty.backward()

    d_loss = loss_fake - loss_real + grad_penalty
    # backprop.
    optimizers['disc'].step()

    if batch_idx % train_config['n_critic'] == 0:
        # ---------------
        # Train generator
        # ---------------
        generator.zero_grad()
        # Re-sample noise
        z.data.normal_(0, 1)
        gen_imgs = generator(z)
        g_loss = discriminator(gen_imgs)
        g_loss = g_loss.mean()
        g_loss.backward(neg_one)
        optimizers['gen'].step()
    else:
        g_loss = None

    return g_loss, d_loss


def run_training(args):
    """Initialize and run the full training process using the hyper-params in args."""
    device, data_loader, train_config, generator, discriminator, optimizers = init_training(args)

    # generate a sample with fixed seed, and reset the seed to pseudo-random
    torch.manual_seed(42)
    z_sample = torch.randn(train_config['batch_size'], train_config['latent_dim'], 1, 1).to(device)
    torch.manual_seed(random.randint(0, 1e10))

    # Loss function for DCGAN
    if args.gan_type == 'dcgan':
        loss_fn = torch.nn.BCELoss().to(device)

    # Training
    for epoch in range(train_config['start_epoch'], train_config['max_epoch']+1):
        for batch_idx, batch in enumerate(data_loader):

            if args.gan_type == 'dcgan':
                g_loss, d_loss = \
                    training_step_dcgan(batch, device, generator, discriminator, optimizers, train_config, loss_fn)
            elif args.gan_type == 'wgan_gp':
                _, d_loss = \
                    training_step_wgan_gp(batch_idx, batch, device, train_config, generator, discriminator, optimizers)
                if _ is not None:
                    g_loss = _

        print('\nEpoch {}/{}:\n'
              '  Discriminator loss={:.4f}\n'
              '  Generator loss={:.4f}'.format(epoch, train_config['max_epoch'], d_loss.item(), g_loss.item()))

        if epoch == 1 or epoch % train_config['sample_save_freq'] == 0:
            # Save sample
            gen_sample = generator(z_sample)
            save_image_grid(
                img_batch=gen_sample[:train_config['grid_size'] ** 2].detach().cpu().numpy(),
                grid_size=train_config['grid_size'],
                epoch=epoch,
                img_path=os.path.join(args.checkpoint_path, 'samples', 'checkpoint_ep{}_sample.png'.format(epoch))
            )
            print('Image sample saved.')

        if epoch == 1 or epoch % train_config['save_freq'] == 0:
            # Save checkpoint
            gen_path = os.path.join(args.checkpoint_path, 'weights', 'checkpoint_ep{}_gen.pt'.format(epoch))
            disc_path = os.path.join(args.checkpoint_path, 'weights', 'checkpoint_ep{}_disc.pt'.format(epoch))
            torch.save(generator.state_dict(), gen_path)
            torch.save(discriminator.state_dict(), disc_path)
            print('Checkpoint.')


def get_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run GAN training.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-g', '--gan_type', type=str, help='Network type.', required=True, choices=['dcgan', 'wgan_gp'])
    parser.add_argument('-d', '--dataset', type=str, help='Dataset to use.', required=True, choices=['cats', 'flowers'])
    parser.add_argument('-p', '--data_path', type=str, help='Download path for the data.', default='./data')
    parser.add_argument('-c', '--checkpoint_path', type=str, default='./checkpoints',
                        help='Save path for checkpoints and samples during training')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    run_training(args)
