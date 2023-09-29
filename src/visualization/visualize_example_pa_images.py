import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from mpl_toolkits.axes_grid1 import ImageGrid
from src.visualization import col_bar

from src.utils.normalizations import range_normalization

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = "NewComputerModern10"
plt.rcParams["font.size"] = 14

example_real_image = "/home/kris/Work/Data/DA_results/Ablation_Study/PAI/Domain_labels_as_conditioning/cINN/2023_02_26_19_17_43/testing/generated_image_data/test_batch_88.npz"

worst, best, repr = 140, 142, 88

# For Gan_cINN
base_path = "/home/kris/Work/Data/DA_results/Ablation_Study/PAI/Domain_labels_as_conditioning/cINN/2023_02_26_19_17_43/testing/generated_image_data"
worst_case = os.path.join(base_path, f"test_batch_{worst}.npz")
best_case = os.path.join(base_path, f"test_batch_{best}.npz")
# representative_case = os.path.join(base_path, f"test_batch_{0}.npz")
representative_case = os.path.join(base_path, f"test_batch_{repr}.npz")

# For UNIT
# base_path = "/home/kris/Work/Data/DA_results/miccai/domain_adaptation_results/unit/2023_02_18_15_05_06/testing/generated_image_data"
# worst_case = os.path.join(base_path, f"test_batch_{23}.npz")
# best_case = os.path.join(base_path, f"test_batch_{31}.npz")
# representative_case = os.path.join(base_path, f"test_batch_{21}.npz")

sim = np.load(best_case)["images_a"].squeeze()
sim = range_normalization(sim[11, :, :])
sim2real = np.load(best_case)["images_ab"].squeeze()
sim2real = range_normalization(sim2real[11, :, :])
real = np.load(representative_case)["images_b"].squeeze()
real = range_normalization(real[11, :, :])

img = plt.imshow(sim)
scale_bar = ScaleBar(0.15625, units="mm", location="lower left")
plt.gca().add_artist(scale_bar)
col_bar(img)
plt.savefig("/home/kris/Work/Data/domain_adaptation_simulations/HSI_Data/figures/PAI_comparison_sim.svg")
# plt.show()
plt.close()

img = plt.imshow(sim2real)
scale_bar = ScaleBar(0.15625, units="mm", location="lower left")
plt.gca().add_artist(scale_bar)
col_bar(img)
plt.savefig("/home/kris/Work/Data/domain_adaptation_simulations/HSI_Data/figures/PAI_comparison_sim2real.svg")
# plt.show()
plt.close()

img = plt.imshow(real)
scale_bar = ScaleBar(0.15625, units="mm", location="lower left")
plt.gca().add_artist(scale_bar)
col_bar(img)
plt.savefig("/home/kris/Work/Data/domain_adaptation_simulations/HSI_Data/figures/PAI_comparison_real.svg")
# plt.show()
plt.close()




# fig = plt.figure(figsize=(10, 8))
# img_grid = ImageGrid(fig, rect=111, nrows_ncols=(3, 3), axes_pad=0.3, cbar_location="right", cbar_mode="single", share_all=True)
#
# for i, (case_path, case_name) in enumerate(zip([best_case, worst_case, representative_case],
#                                                ["best_case", "worst_case", "representative_case"])):
#
#         data = np.load(case_path)["images_a"].squeeze()
#         data = range_normalization(np.sqrt(data[9, :, :]))
#         img_grid[i].imshow(data)
#         img_grid[i].set_title(f"Simulated {case_name}")
#         img_grid[i].axis("off")
#
#         scale_bar = ScaleBar(0.15625, units="mm", location="lower left",
#                              font_properties={"family": "Libertinus Serif", "size": 10})
#         img_grid[i].add_artist(scale_bar)
#
#         data = np.load(case_path)["images_ab"].squeeze()
#         data = range_normalization(data[9, :, :])
#         img_grid[i + 3].imshow(data)
#         img_grid[i + 3].set_title(f"Translated {case_name}")
#
#         img_grid[i + 3].axis("off")
#
#         scale_bar = ScaleBar(0.15625, units="mm", location="lower left",
#                              font_properties={"family": "Libertinus Serif", "size": 10})
#         img_grid[i + 3].add_artist(scale_bar)
#
#         data = np.load(case_path)["images_b"].squeeze()
#         data = range_normalization(np.sqrt(data[9, :, :]))
#         real_im = img_grid[i + 6].imshow(data)
#         img_grid[i + 6].set_title(f"Exemplary Real Image")
#
#         img_grid[i + 6].axis("off")
#
#         scale_bar = ScaleBar(0.15625, units="mm", location="lower left",
#                              font_properties={"family": "Libertinus Serif", "size": 10})
#         img_grid[i + 6].add_artist(scale_bar)
#
#         cb = img_grid.cbar_axes[0].colorbar(real_im)
#         cb.set_ticks((0, 1))
#
# plt.tight_layout()
# plt.show()
