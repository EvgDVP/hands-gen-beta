import torch
from torchvision.utils import save_image
import os

# Параметры генерации
latent_dim = 100  # размер латентного вектора (в зависимости от вашей модели)
num_images = 10  # количество изображений, которые нужно сгенерировать
output_dir = './generated_images'  # директория для сохранения изображений
os.makedirs(output_dir, exist_ok=True)

# Загрузка модели генератора
class Generator(torch.nn.Module):
    def __init__(self, latent_dim, condition_dim):
        super(Generator, self).__init__()

        self.init_size = 4  # Начальный размер изображения 4x4
        self.l1 = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 1024 * self.init_size * self.init_size)
        )

        self.conv_blocks = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 128x128 -> 256x256
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            # 256x256 -> 512x512
            nn.ConvTranspose2d(16, 3, 4, 2, 1),
            nn.Tanh()  # Для нормализации значений пикселей в диапазоне от -1 до 1
        )

    def forward(self, z, condition):
        # Конкатенируем латентный вектор и условие
        x = torch.cat([z, condition], dim=1)

        # Преобразуем через полносвязный слой в начальный тензор
        out = self.l1(x)
        out = out.view(out.size(0), 1024, self.init_size, self.init_size)  # Превращаем в тензор 4x4

        # Пропускаем через транспонированные свёртки для увеличения разрешения
        img = self.conv_blocks(out)
        return img

# Инициализация модели генератора
generator = Generator()

# Загрузка сохранённого веса модели
generator.load_state_dict(torch.load('generator.pth'))
generator.eval()  # Перевод в режим генерации (оценки)

# Генерация изображений
def generate_images(num_images):
    # Генерация случайного шума (латентный вектор) для входа в генератор
    noise = torch.randn(num_images, latent_dim)

    # Прогон через генератор
    with torch.no_grad():  # Отключаем вычисление градиентов для генерации
        generated_images = generator(noise)

    return generated_images

# Сохранение изображений
def save_generated_images(images, output_dir):
    for i, img in enumerate(images):
        save_image(img, os.path.join(output_dir, f'image_{i+1}.png'))

# Генерация и сохранение изображений
images = generate_images(num_images)
save_generated_images(images, output_dir)

print(f'Сгенерированные {num_images} изображения были сохранены в {output_dir}')