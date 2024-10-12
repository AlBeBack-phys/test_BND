import cv2
import torch
import torchvision
import numpy as np

def segment_people(video_path, output_path, batch_size=5, threshold=0.3, mask_alpha=0.4, class_scores=True):
    """Выделяет людей на видео по пути video_path и сохраняет результат в output path.
    Каждая отрисовка содержит имя класса и уверенность модели.

    Для сегментации используется архитектура Mask R-CNN

    Поддерживает обработку видео как на CPU, так и на GPU. Рекомендуется обрабатывать видео с помощью GPU - это
    существенно ускорит процесс.
    Функция использует GPU автоматически, если оно доступно.
    Кадры обрабатываются батчами для ускорения работы кода. Размер батча можно регулировать.

    Args:
        video_path (str): Путь к входному видеофайлу.
        output_path (str): Путь к выходному видеофайлу.
        batch_size (int): Размер батча для обработки кадров.
        threshold (float): Порог уверенности для предсказаний модели. Если уверенность < threshold, данный объект
        отрисовываться не будет
        mask_alpha (float): Прозрачность маски при наложении.
        class_scores(bool): Указывает, отрисовывать ли класс и уверенность модели
    Returns:
        None
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #Загружаем испольлзуемую для сегментации модель
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()

    # Преобразование кадров в тензоры
    transform = torchvision.transforms.ToTensor()

    def detect_and_draw_batch(frames_batch, threshold, mask_alpha, class_scores):
        """Обрабатывает кадры в батчах и выполняет сегментацию.

        Args:
            frames_batch (list): Список кадров для обработки.
            threshold (float): Порог уверенности для предсказаний модели. Если уверенность < threshold, данный объект
            отрисовываться не будет
            mask_alpha (float): Прозрачность маски при наложении.
            class_scores(bool): Указывает, отрисовывать ли класс и уверенность модели

        Returns:
            list: Список обработанных кадров с наложенными масками.
        """
        input_tensors = torch.stack([transform(frame) for frame in frames_batch]).to(device)

        with torch.no_grad():
            predictions = model(input_tensors)  # Предсказание для батча кадров

        result_frames = []
        for frame, prediction in zip(frames_batch, predictions):
            filtered_boxes = []
            for i in range(len(prediction['masks'])):
                mask = prediction['masks'][i, 0].mul(255).byte().cpu().numpy()
                label = prediction['labels'][i]
                score = prediction['scores'][i]

                if score > threshold and label == 1:  # Устанавливаем порог уверенности и нужный нам класс(Person)
                    box = prediction['boxes'][i]

                    color = (0, 255, 0)  # Зелёный цвет

                    filtered_boxes.append(box)

                    colored_mask = np.zeros_like(frame)
                    colored_mask[mask > 127] = color

                    # Наложение маски на кадр
                    frame = cv2.addWeighted(colored_mask, mask_alpha, frame, 1, 0)

                    # Отрисовка метки и уверенности
                    if class_scores:
                        x1, y1, x2, y2 = box.int().cpu().numpy()
                        cv2.putText(frame, f"Person: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            result_frames.append(frame)

        return result_frames

    def save_video(output_path, frames, fps=30):
        """Сохраняет обработанные кадры в видеофайл.

        Args:
            output_path (str): Путь к выходному видеофайлу.
            frames (list): Список кадров для сохранения.
            fps (int): Частота кадров видео.
        """
        height, width, _ = frames[0].shape
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

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
                frames.extend(detect_and_draw_batch(batch, threshold, mask_alpha, class_scores))  # Обработка последнего неполного батча
            break

        batch.append(frame)

        if len(batch) == batch_size:
            frames.extend(detect_and_draw_batch(batch, threshold, mask_alpha, class_scores))
            batch = []

    cap.release()
    save_video(output_path, frames)


if __name__ == '__main__':
    try:
        video_path = input('Введите путь к видео, которое вы хотите обработать: ')
        output_path = input('Укажите путь, куда нужно сохранить результат: ')
        segment_people(video_path, output_path)
    except Exception as e:
        print(f"Ошибка: {e}")
