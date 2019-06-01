import json
import argparse
import os
import os.path as osp
import torch
from torch.utils.data import DataLoader
import utils
import cat_dataset
import networks


def run_training():
    if params.dataset == "cats":
        # download and extract data
        img_paths, annotation_paths = cat_dataset.prepare_dataset(data_path=params.data_path)
        catface_img_paths = cat_dataset.extract_catfaces(img_paths, annotation_paths)
        # define dataset
        dataset = cat_dataset.CatfaceDataset(img_paths=catface_img_paths, mirror=True, random_crop=True)
        print("num of images:", len(dataset))

    # load training configuration
    if params.gan_type == "dcgan":
        with open('dcgan_train_config.json', 'r') as json_file:
            train_config = json.load(json_file)

    elif params.gan_type == "wgan_gp":
        with open('wgan_gp_train_config.json', 'r') as json_file:
            train_config = json.load(json_file)

    # define data loader
    data_loader = DataLoader(dataset, batch_size=train_config["batch_size"],
                             shuffle=True, num_workers=train_config["num_workers"])

    # use cuda if available
    use_gpu = torch.cuda.is_available()
    print("Use GPU: {}".format(use_gpu))

    # Initialize generator and discriminator
    generator = networks.Generator()  # the generator is the same for both the DCGAN and the WGAN

    if params.gan_type == "dcgan":
        discriminator = networks.DCGAN_Discriminator()  # DCGAN discriminator
        # Loss function
        adversarial_loss = torch.nn.BCELoss()
        if use_gpu:
            adversarial_loss = adversarial_loss.cuda()

    elif params.gan_type == "wgan_gp":
        discriminator = networks.WGAN_Discriminator()  # WGAN Discriminator

    if use_gpu:
        generator = generator.cuda()
        discriminator = discriminator.cuda()

    # Initialize weights
    generator.apply(networks.weights_init)
    discriminator.apply(networks.weights_init)

    # Optimizers
    optimizer_gen = torch.optim.Adam(generator.parameters(), lr=train_config["learning_rate_g"],
                                     betas=(train_config["b1"], train_config["b2"]))
    optimizer_disc = torch.optim.Adam(discriminator.parameters(), lr=train_config["learning_rate_d"],
                                      betas=(train_config["b1"], train_config["b2"]))

    # make save dir, if needed
    os.makedirs(osp.join(params.checkpoint_path, "weights"), exist_ok=True)
    os.makedirs(osp.join(params.checkpoint_path, "samples"), exist_ok=True)

    # generate a sample
    z_sample = torch.randn(train_config["batch_size"], train_config["latent_dim"], 1, 1)
    if use_gpu:
        z_sample = z_sample.cuda()

    # load weights if the training is not starting from the beginning
    if train_config["start_epoch"] > 1:
        device = "cuda:0" if use_gpu else "cpu"
        generator.load_state_dict(torch.load(osp.join(params.checkpoint_path, "weights", "checkpoint_ep{}_gen.pt"
                                                      .format(train_config["start_epoch"]-1)), map_location=device))
        discriminator.load_state_dict(torch.load(osp.join(params.checkpoint_path, "weights", "checkpoint_ep{}_disc.pt"
                                                          .format(train_config["start_epoch"]-1)), map_location=device))

    # start training
    for epoch in range(train_config["start_epoch"], train_config["max_epoch"]+1):
        for batch_idx, imgs in enumerate(data_loader):
            if use_gpu:
                imgs = imgs.cuda()

            # Sample noise as generator input
            z = torch.randn(imgs.size()[0], train_config["latent_dim"], 1, 1)
            if use_gpu:
                z = z.cuda()

            if params.gan_type == "dcgan":
                valid = torch.ones(imgs.size(0))
                fake = torch.zeros(imgs.size(0))
                if use_gpu:
                    valid, fake = valid.cuda(), fake.cuda()

                # -------------------
                # Train Discriminator
                # -------------------
                optimizer_disc.zero_grad()
                # Sample real
                real_loss = adversarial_loss(discriminator(imgs), valid)
                # Sample fake
                gen_imgs = generator(z)  # generate fakes
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                # Backpropagate
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_disc.step()

                # ---------------
                # Train generator
                # ---------------
                optimizer_gen.zero_grad()
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)
                g_loss.backward()
                optimizer_gen.step()

            elif params.gan_type == "wgan_gp":
                one, neg_one = torch.FloatTensor([1]), torch.FloatTensor([1])*-1
                u = torch.FloatTensor(imgs.size()[0], 1, 1, 1).uniform_(0, 1)
                grad_outputs = torch.ones(imgs.size()[0])
                if use_gpu:
                    one, neg_one = one.cuda(), neg_one.cuda()
                    u = u.cuda()
                    grad_outputs = grad_outputs.cuda()

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
                interpolates = u*imgs + (1-u)*gen_imgs
                if use_gpu:
                    interpolates = interpolates.cuda()
                interpolates.requires_grad_(True)
                grad = torch.autograd.grad(outputs=discriminator(interpolates), inputs=interpolates,
                                           grad_outputs=grad_outputs, retain_graph=True,
                                           create_graph=True, only_inputs=True)[0]
                grad_penalty = train_config["grad_penalty_weight"] * \
                    ((grad.norm(2, 1).norm(2, 1).norm(2, 1) - 1) ** 2).mean()
                grad_penalty.backward()
                # Optimize
                d_loss = loss_fake - loss_real + grad_penalty
                optimizer_disc.step()

                if batch_idx % train_config["n_critic"] == 0:
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
                    optimizer_gen.step()

        print("\nEpoch {}/{}:\n"
              "  Discriminator loss={:.4f}\n"
              "  Generator loss={:.4f}".format(epoch, train_config["max_epoch"], d_loss.item(), g_loss.item()))

        if epoch == 1 or epoch % train_config["sample_save_freq"] == 0:
            # Save sample
            gen_sample = generator(z_sample)
            utils.save_image_grid(img_batch=gen_sample[:train_config["grid_size"] ** 2].detach().cpu().numpy(),
                                  grid_size=train_config["grid_size"], epoch=epoch,
                                  img_path=osp.join(params.checkpoint_path, "samples",
                                                    "checkpoint_ep{}_sample.png".format(epoch)))
            print("Image sample saved.")

        if epoch == 1 or epoch % train_config["save_freq"] == 0:
            torch.save(generator.state_dict(), osp.join(params.checkpoint_path, "weights",
                                                        "checkpoint_ep{}_gen.pt".format(epoch)))
            torch.save(discriminator.state_dict(), osp.join(params.checkpoint_path, "weights",
                                                            "checkpoint_ep{}_disc.pt".format(epoch)))
            print("Checkpoint.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GAN training.")
    parser.add_argument("--gan_type", type=str, help="network type", required=True, choices=["dcgan", "wgan_gp"])
    parser.add_argument("--dataset", type=str, help="dataset for training", required=True, choices=["cats", "flowers"])
    parser.add_argument("--data_path", type=str, help="download path for the data", default="./data")
    parser.add_argument("--checkpoint_path", type=str, help="save path for checkpoints and samples during training",
                        default="./checkpoints")
    params = parser.parse_args()

    run_training()
