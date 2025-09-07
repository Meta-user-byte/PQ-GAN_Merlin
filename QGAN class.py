class QGAN:
    def __init__(self, image_size, gen_count, gen_arch, input_state, noise_dim, batch_size, pnr, lossy):
        self.image_size = image_size
        self.gen_count = gen_count
        self.noise_dim = max(1, int(noise_dim))
        self.latent_dim = self.noise_dim * max(1, int(gen_count))
        self.batch_size = int(batch_size)

        # Generator & Discriminator
        self.G = MerlinPatchGenerator(
            image_size=image_size,
            gen_count=gen_count,
            gen_arch=gen_arch,
            input_state=pcvl.BasicState(input_state) if not hasattr(input_state, "m") else input_state,
            noise_dim=noise_dim,
            pnr=pnr,
            lossy=lossy,
            shots=3000,
            use_clements=False,
        )
        self.D = MerlinDiscriminator(image_size=image_size)  # logits

        # Opts (set during fit)
        self.optG = None
        self.optD = None

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.G.to(self.device)
        self.D.to(self.device)

        #Warm-up once to initialize any Lazy* params BEFORE counting 
        with torch.no_grad():
            _ = self.G(torch.zeros(1, self.latent_dim, device=self.device))
            _ = self.D(torch.zeros(1, self.image_size * self.image_size, device=self.device))

        #  Now it's safe to count; also guard against any leftover uninitialized 
        def _count_params(m):
            total = 0
            for p in m.parameters():
                if not getattr(p, "requires_grad", False):
                    continue
                try:
                    total += p.numel()
                except Exception:
                    # UninitializedParameter or similar; skip counting
                    continue
            return total

        print(f"Generator trainable params: {_count_params(self.G)}")
        print(f"Discriminator trainable params: {_count_params(self.D)}")

    # always return a non-empty trainable vector for logging
    def _get_trainable_vector(self):
        """
        Priority:
          1) Concatenate theta vectors from each quantum sublayer if they exist.
          2) Else, concatenate all self.G.parameters() that require grad.
          3) Else, return a tiny dummy so CSV is never empty.
        """
        vecs = []
        if hasattr(self.G, "layers"):
            for layer in self.G.layers:
                t = getattr(layer, "theta", None)
                if isinstance(t, torch.Tensor):
                    vecs.append(t.view(-1))
        if vecs:
            return torch.cat(vecs)

        params = [p.view(-1) for p in self.G.parameters() if isinstance(p, torch.Tensor) and p.requires_grad]
        if params:
            return torch.cat(params)

        return torch.zeros(1, device=self.device)

    def fit(self, dataloader, lrD, opt_params, silent=False):

        def _save_grid(batch_flat: torch.Tensor, step: int, outdir: str = "samples"):
            N = batch_flat.shape[0]
            H = W = self.image_size

            imgs01 = ((batch_flat.clamp(-1,1) + 1.0)/2.0).detach().cpu().view(N,H,W).numpy()

            ncols = min(8, N)
            nrows = math.ceil(N/ ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.2, nrows * 1.2))
            axes = np.atleast_1d(axes).ravel()
            for i in range(nrows * ncols):
                ax = axes[i]
                ax.axis("off")
                if i < N:
                    ax.imshow((imgs01[i] * 255).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
            os.makedirs(outdir, exist_ok=True)
            fig.savefig(os.path.join(outdir, f"step_{step:04d}.png"), dpi=150, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
        # Sanity: latent size must match
        expected = self.noise_dim * self.gen_count
        assert self.latent_dim == expected, f"[Sanity] latent_dim mismatch: {self.latent_dim} vs {expected}"

        # --- TTUR tweak: weaken D a tad, strengthen G a tad ---
        self.optD = torch.optim.Adam(self.D.parameters(), lr=lrD * 0.08, betas=(0.5, 0.999))
        self.optG = torch.optim.Adam(self.G.parameters(), lr=lrD * 1.00, betas=(0.5, 0.999))

        # Make Merlin deterministic (no shot noise)
        try:
            self.G.sample_count = 1
            if hasattr(self.G, "sampler"):
                self.G.sampler._iterator = []
        except Exception:
            pass

        total_steps = int(opt_params.get("opt_iter_num", 1500))
        steps_done = 0

        D_loss_progress = []
        G_loss_progress = []
        G_params_progress = []
        fake_data_progress = []

        pbar = dataloader if silent else tqdm(
            dataloader, total=total_steps, desc="Merlin-QGAN training", leave=False
        )

        last_loss_G = None
        last_gen_for_log = None

        #Sanity trackers
        grad_nonzero_steps = 0

        # Fixed probe to check G changes over time
        z_probe = torch.randn(self.batch_size, self.latent_dim, device=self.device)
        with torch.no_grad():
            probe_out_prev = self.G(z_probe).detach()

        first_logged = False
        r1_gamma = 3e-3  # tiny, stable

        for batch in pbar:
            # === Prepare real data ===
            real_imgs = batch[0] if isinstance(batch, (list, tuple)) else batch
            if real_imgs.dim() == 3:
                real_imgs = real_imgs.unsqueeze(1)  # (B,1,8,8)
            real_imgs = real_imgs.to(self.device).float()

            # --- Normalize to [0,1] if needed ---
            with torch.no_grad():
                if real_imgs.max() > 1.0:
                    real_imgs.div_(255.0)  

            real_flat = real_imgs.view(real_imgs.size(0), -1)  # (B,64)
            B = real_flat.size(0)

            # Instance noise on real data for D
            real_noisy = real_flat + 0.05 * torch.randn_like(real_flat)
            frac = steps_done / float(total_steps)
            sigma = 0.05 * (1.0 - frac) + 0 * frac

            # (One-time) check D logits at start
            if not first_logged:
                with torch.no_grad():
                    z0 = torch.randn(B, self.latent_dim, device=self.device)
                    f0 = self.G(z0)
                    d_real0 = self.D(real_flat).mean().item()
                    d_fake0 = self.D(f0).mean().item()
                    print(f"[Sanity] D logits at start: real={d_real0:.3f}, fake={d_fake0:.3f}")
                first_logged = True

            # === Train D (non-sat loss + tiny R1 on real) ===
            self.optD.zero_grad()

            real_noisy.requires_grad_(True)
            real_logits = self.D(real_noisy)
            if real_logits.dim() == 1:
                real_logits = real_logits.unsqueeze(1)

            z = torch.randn(B, self.latent_dim, device=self.device)
            fake_flat = self.G(z).detach()
            if not torch.isfinite(fake_flat).all():
                print("[Sanity] Non-finite in generator output; clamping.")
                fake_flat = torch.nan_to_num(fake_flat, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

            fake_logits = self.D(fake_flat)
            if fake_logits.dim() == 1:
                fake_logits = fake_logits.unsqueeze(1)


            fake_noisy = fake_flat + 0.05 * torch.randn_like(fake_flat)
            fake_logits = self.D(fake_noisy)

            # Non-saturating D loss
            loss_D_main = F.softplus(-real_logits).mean() + F.softplus(fake_logits).mean()

            # R1 on real
            grad_real = torch.autograd.grad(
                outputs=real_logits.sum(), inputs=real_noisy, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
            r1_penalty = (grad_real.pow(2).sum(dim=1).mean()) * (r1_gamma * 0.5)

            loss_D = loss_D_main + r1_penalty
            loss_D.backward()
            self.optD.step()

            # === Train G (every step) ===
            self.optG.zero_grad()
            z = torch.randn(B, self.latent_dim, device=self.device)
            gen_flat = self.G(z)

            if not torch.isfinite(gen_flat).all():
                print("[Sanity] Non-finite in generator output (G step); clamping.")
                gen_flat = torch.nan_to_num(gen_flat, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

            gen_logits = self.D(gen_flat)
            if gen_logits.dim() == 1:
                gen_logits = gen_logits.unsqueeze(1)

            # Non-saturating G loss
            loss_G = F.softplus(-gen_logits).mean()
            loss_G.backward()

            # gradient sanity counter
            nz = 0.0
            for p in self.G.parameters():
                if p.requires_grad and p.grad is not None:
                    nz += p.grad.detach().abs().sum().item()
            if nz > 0:
                grad_nonzero_steps += 1

            torch.nn.utils.clip_grad_norm_(self.G.parameters(), 1.0)
            self.optG.step()
            last_loss_G = float(loss_G.detach().item())
            last_gen_for_log = gen_flat.detach()

            #Output stats / probe drift every 100 steps
            if (steps_done % 5) == 0:
                with torch.no_grad():
                    g_mean = last_gen_for_log.mean().item()
                    g_std  = last_gen_for_log.std().item()
                    probe_now = self.G(z_probe).detach()
                    drift = (probe_now - probe_out_prev).pow(2).mean().sqrt().item()  # RMSE
                print(f"[Sanity] step {steps_done}: gen mean={g_mean:.4f}, std={g_std:.4f}, probe_RMSE={drift:.6f}")
                probe_out_prev = probe_now

            # === Logging ===
            D_loss_progress.append(loss_D.item())
            G_loss_progress.append(last_loss_G)

            with torch.no_grad():
                g_vec = self._get_trainable_vector().detach().float().cpu().numpy()
            G_params_progress.append(g_vec)
            fake_data_progress.append(last_gen_for_log.mean(0).detach().cpu().numpy())

            z_fixed = torch.randn(16, self.latent_dim, device=self.device)

            steps_done += 1
            if steps_done % 300 == 0:
                with torch.no_grad():
                    self.G.eval()
                    grid = self.G(z_fixed)
                    _save_grid(grid, steps_done, outdir="samples")
                    self.G.train()
            if steps_done >= total_steps:
                break

        print(f"[Sanity] Steps with non-zero G gradients: {grad_nonzero_steps}/{steps_done}")

        # Return as lists/arrays (caller writes CSVs)
        return D_loss_progress, G_loss_progress, G_params_progress, np.vstack(fake_data_progress)
