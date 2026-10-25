"""Задание 3. Сравнение производительности CPU и CUDA"""
import time

import torch

#Размеры матриц из задания 3.1: (число матриц, строки, столбцы)
SHAPES = [(64, 1024, 1024), (128, 512, 512), (256, 256, 256)]
CPU_REPEATS = 3  #сколько раз повторяем операцию на CPU (она медленная)
GPU_REPEATS = 10  #на GPU повторяем чаще, т.к. операции быстрые

HAS_CUDA = torch.cuda.is_available()  #есть ли видеокарта с CUDA
DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")


def make_data(shape):
    """Создаёт две случайные матрицы нужной формы (на CPU)"""
    if len(shape) != 3 or shape[1] != shape[2]:  #проверка: нужны 3 размера и квадратные матрицы (для matmul)
        raise ValueError(f"Неподходящая форма: {shape}")
    a = torch.rand(shape)  #случайные числа от 0 до 1, тип float32
    b = torch.rand(shape)
    return a, b


#операции, которые сравниваем. каждая принимает две матрицы a и b
OPERATIONS = {
    "Матричное умножение": lambda a, b: torch.matmul(a, b),
    "Сложение": lambda a, b: a + b,
    "Умножение (поэлем.)": lambda a, b: a * b,
    #transpose сам по себе только меняет вид тензора без копирования данных,
    #поэтому добавляем contiguous(), чтобы данные реально переставились в памяти
    "Транспонирование": lambda a, b: a.transpose(1, 2).contiguous(),
    "Сумма элементов": lambda a, b: a.sum(),
}


def measure_cpu(func, repeats=CPU_REPEATS):
    """Среднее время операции на CPU в миллисекундах (через time.time())"""
    func()  #один пробный запуск (прогрев), его не считаем
    start = time.time()
    for _ in range(repeats):
        func()
    end = time.time()
    return (end - start) / repeats * 1000  #секунды -> миллисекунды, делим на число повторов


def measure_gpu(func, repeats=GPU_REPEATS):
    """Среднее время операции на GPU в миллисекундах (через torch.cuda.Event)"""
    func()  #прогрев: первый запуск на GPU включает инициализацию, его не считаем
    torch.cuda.synchronize()  #ждём, пока GPU закончит все задачи
    start = torch.cuda.Event(enable_timing=True)  #метка времени на самой видеокарте
    end = torch.cuda.Event(enable_timing=True)
    start.record()  #ставим метку старта
    for _ in range(repeats):
        func()
    end.record()  #ставим метку конца
    torch.cuda.synchronize()  #ждём завершения, иначе время будет неверным (GPU работает асинхронно)
    return start.elapsed_time(end) / repeats  # elapsed_time уже в миллисекундах


def measure_transfer(tensor):
    """Время копирования тензора с CPU на GPU в миллисекундах"""
    torch.cuda.synchronize()
    start = time.time()
    tensor.to(DEVICE)
    torch.cuda.synchronize()
    return (time.time() - start) * 1000


def check_correctness():
    """Проверка на маленьком примере: результаты CPU и GPU должны совпадать"""
    a, b = make_data((4, 8, 8))
    for name, op in OPERATIONS.items():
        cpu_result = op(a, b)
        if HAS_CUDA:
            gpu_result = op(a.to(DEVICE), b.to(DEVICE)).cpu()  #считаем на GPU и возвращаем на CPU
            assert torch.allclose(cpu_result, gpu_result, rtol=1e-3, atol=1e-3), f"{name}: CPU и GPU не совпали"
    assert torch.matmul(torch.eye(3), torch.ones(3, 3)).sum() == 9  #единичная матрица не меняет единичную: сумма 9
    print("Проверка корректности пройдена\n")


def run_benchmark():
    """Для каждого размера замеряет все операции и печатает таблицу"""
    for shape in SHAPES:
        a, b = make_data(shape)
        size_mb = a.numel() * a.element_size() / 1024 ** 2  #размер одной матрицы в мегабайтах
        print(f"Размер {shape}, одна матрица = {size_mb:.0f} МБ")
        print(f"{'Операция':<22}| {'CPU (мс)':>10} | {'GPU (мс)':>10} | {'Ускорение':>10}")
        print("-" * 62)

        if HAS_CUDA:
            a_gpu, b_gpu = a.to(DEVICE), b.to(DEVICE)  #переносим данные на видеокарту один раз

        for name, op in OPERATIONS.items():
            cpu_ms = measure_cpu(lambda: op(a, b))
            if HAS_CUDA:
                gpu_ms = measure_gpu(lambda: op(a_gpu, b_gpu))
                speedup = cpu_ms / gpu_ms  #во сколько раз GPU быстрее
                print(f"{name:<22}| {cpu_ms:>10.2f} | {gpu_ms:>10.2f} | {speedup:>9.1f}x")
            else:
                print(f"{name:<22}| {cpu_ms:>10.2f} | {'-':>10} | {'-':>10}")

        if HAS_CUDA:
            print(f"Копирование одной матрицы CPU -> GPU: {measure_transfer(a):.2f} мс")
            del a_gpu, b_gpu
            torch.cuda.empty_cache()  #освобождаем память видеокарты перед следующим размером
        print()
        del a, b


ANALYSIS = """
Анализ результатов:
1) Наибольшее ускорение обычно даёт матричное умножение: в нём очень много вычислений на каждый
   элемент данных, и тысячи ядер GPU работают параллельно. Поэлементные операции (сложение,
   умножение), транспонирование и сумма упираются в скорость памяти, а не в вычисления, поэтому
   ускорение у них меньше
2) Часть операций может быть медленнее на GPU или почти не быстрее, если данных мало: на запуск
   каждой операции на GPU уходит фиксированное время (запуск ядра, синхронизация), и оно может
   быть больше самой работы. Ещё первый запуск на GPU включает инициализацию (поэтому есть прогрев)
3) Чем больше матрицы, тем больше ускорение: фиксированные расходы на запуск делятся на большую
   работу, а GPU лучше загружается. На маленьких матрицах ускорение падает
4) Передача данных CPU <-> GPU идёт по шине PCIe, она намного медленнее памяти самой видеокарты.
   Копирование большой матрицы может занимать столько же времени, сколько сама операция, а то и
   больше. Поэтому одну простую операцию невыгодно отдавать на GPU вместе с копированием: перенос
   выгоден, когда данные один раз переносятся и над ними выполняется много операций
"""


if __name__ == "__main__":
    print("torch:", torch.__version__)
    print("CUDA доступна:", HAS_CUDA)
    if HAS_CUDA:
        print("Видеокарта:", torch.cuda.get_device_name(0))
    else:
        print("GPU не найден, замеряем только CPU")
    print()
    check_correctness()
    run_benchmark()
    print(ANALYSIS)
