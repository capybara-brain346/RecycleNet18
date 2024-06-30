from torch.utils.data import DataLoader
from custom_data_loader import ImageLoader
from config import Config


class ImageNormalization:
    def __init__(self, dataset):
        self.dataset = dataset
        self.loader = DataLoader(self.dataset, batch_size=64, shuffle=False)
        self.mean, self.std, self.nb_samples = 0.0, 0.0, 0.0

    def normalize(self):
        for data in self.loader:
            data = data[0]
            batch_samples = data.size(0)
            data = data.view(batch_samples, data.size(1), -1)
            self.mean += data.mean(2).sum(0)
            self.std += data.std(2).sum(0)
            self.nb_samples += batch_samples

        self.mean /= self.nb_samples
        self.std /= self.nb_samples

        return self.mean, self.std


if __name__ == "__main__":
    path = Config.DATA_BASE_DIR
    data = ImageLoader(path=path)
    normalize = ImageNormalization(dataset=data)
    print(normalize.normalize())
