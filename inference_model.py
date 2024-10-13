import random
import cv2
import torch
import torchvision
import numpy as np


def generate_random_color():
    """
    Генерирует случайный цвет.

    Returns:
        tuple: Цвет в формате (R, G, B).
    """
    return tuple(random.randint(0, 255) for _ in range(3))


def calculate_iou(box1, box2):
    """
    Вычисляет коэффициент Intersection over Union (IoU) для двух прямоугольных боксов.

    Args:
        box1 (list or tuple): Координаты первого бокса в формате [x1, y1, x2, y2].
        box2 (list or tuple): Координаты второго бокса в формате [x1, y1, x2, y2].

    Returns:
        float: Коэффициент IoU.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


class ObjectTracker:
    """
    Класс, который осуществляет трекинг объектов для данного видео, определяя, какому боксу из предыдущего кадра
    соответствует каждый бокс из текущего кадра. Если бокса, соответствующего данному, в прошлом кадре определить не удалось,
    данный бокс воспринимается как новый, ему присваивается случайный цвет.

    Attributes:
        previous_colors (dict): Словарь для хранения цветов объектов с предыдущего кадра.
    """

    def __init__(self):
        """Инициализация трекера объектов."""
        self.previous_colors = {}  # Хранит цвет боксов для предыдущего кадра

    def track_objects(self, current_boxes):
        """
        Отслеживает объекты и назначает им цвета на основе максимального значения IoU.

        Args:
            current_boxes (list): Список координат боксов с текущего кадра.

        Returns:
            dict: Словарь с боксами и их назначенными цветами.
        """
        current_colors = {}

        for curr_box in current_boxes:
            best_iou = 0
            best_object_id = None

            # Находим бокс, у которого максимальный IoU с данным
            for prev_box in self.previous_colors:
                iou = calculate_iou(curr_box, prev_box)
                if iou > best_iou:
                    best_iou = iou
                    best_object_id = prev_box

            if best_object_id is not None and best_iou > 0.1:
                object_id = tuple(best_object_id)
                color = self.previous_colors.get(object_id)
                if color is None:
                    color = generate_random_color()
            else:
                color = generate_random_color()
            current_colors[tuple(curr_box)] = color

        # Обновляем цвета для текущего кадра
        self.previous_colors = current_colors
        return current_colors


def segment_people(video_path, output_path, batch_size=5, threshold=0.3, mask_alpha=0.6):
    """
    Выделяет людей на видео по пути video_path и сохраняет результат в output_path.
    Каждая отрисовка содержит имя класса и уверенность модели.

    Для сегментации используется архитектура Mask R-CNN

    Поддерживает обработку видео как на CPU, так и на GPU. Рекомендуется обрабатывать видео с помощью GPU - это
    существенно ускорит процесс.
    Функция использует GPU автоматически, если оно доступно.
    Кадры обрабатываются батчами для ускорения работы кода(в случае если обработка происходит на GPU). Размер батча можно регулировать.

    Args:
        video_path (str): Путь к входному видеофайлу.
        output_path (str): Путь к выходному видеофайлу.
        batch_size (int): Размер батча для обработки кадров.
        threshold (float): Порог уверенности для предсказаний модели. Если уверенность < threshold, данный объект
        отрисовываться не будет
        mask_alpha (float): Прозрачность маски при наложении.
        class_scores(bool): Указывает, отрисовывать ли класс и уверенность модели
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Загружаем модель Mask R-CNN
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    transform = torchvision.transforms.ToTensor()
    tracker = ObjectTracker()

    def detect_and_track(frames_batch):
        """
        Выполняет детекцию и трекинг объектов на кадрах видео.

        Args:
            frames_batch (list): Список кадров видео.

        Returns:
            list: Список кадров с нанесёнными масками объектов.
        """
        input_tensors = torch.stack([transform(frame) for frame in frames_batch]).to(device)

        with torch.no_grad():
            predictions = model(input_tensors)

        result_frames = []
        for frame, prediction in zip(frames_batch, predictions):
            current_boxes = []

            for i in range(len(prediction['masks'])):
                score = prediction['scores'][i]
                if score > threshold and prediction['labels'][i] == 1:  # Класс 1 - человек
                    box = prediction['boxes'][i]
                    current_boxes.append(box.int().cpu().numpy())

            # Отслеживаем объекты и назначаем цвета
            object_colors = tracker.track_objects(current_boxes)

            for i in range(len(prediction['masks'])):
                if prediction['scores'][i] > threshold and prediction['labels'][i] == 1:
                    mask = prediction['masks'][i, 0].mul(255).byte().cpu().numpy()
                    box = prediction['boxes'][i]
                    object_id = tuple(box.int().cpu().numpy())

                    color = object_colors.get(object_id, (0, 0, 0))

                    colored_mask = np.zeros_like(frame)
                    colored_mask[mask > 127] = color

                    # Наносим маску на изображение
                    frame = cv2.addWeighted(colored_mask, mask_alpha, frame, 1.0, 0)

                    # Отрисовка класса и уверенности
                    x1, y1, x2, y2 = box.int().cpu().numpy()
                    cv2.putText(frame, f"Person: {prediction['scores'][i]:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            result_frames.append(frame)

        return result_frames

    def save_video(output_path, frames, fps=30):
        """
        Сохраняет видео на диск.

        Args:
            output_path (str): Путь для сохранения видео.
            frames (list): Список обработанных кадров.
            fps (int, optional): Количество кадров в секунду. По умолчанию 30.
        """
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        for frame in frames:
            out.write(frame)
        out.release()

    cap = cv2.VideoCapture(video_path)
    frames = []
    batch = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            if batch:
                frames.extend(detect_and_track(batch))
            break

        batch.append(frame)

        if len(batch) == batch_size:
            frames.extend(detect_and_track(batch))
            batch = []

    cap.release()
    save_video(output_path, frames)


if __name__ == '__main__':
    video_path = input('Введите путь к видео: ')
    output_path = input('Укажите путь для сохранения результата: ')
    segment_people(video_path, output_path)
