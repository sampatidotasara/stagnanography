from dataset.div2k_dataset import DIV2KDataset

dataset = DIV2KDataset(
    image_dir="data/DIV2K_train_HR",
    image_size=256
)

print("Dataset Size:", len(dataset))

cover, secret = dataset[0]

print("Cover Shape :", cover.shape)
print("Secret Shape:", secret.shape)
