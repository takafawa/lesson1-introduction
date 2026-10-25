"""Задание 1. Создание и манипуляции с тензорами"""
import math

import torch

torch.manual_seed(42)  #случайность, чтобы числа были одинаковыми при каждом запуске


def create_tensors():
    """Создаёт тензоры из пункта 1.1 и возвращает их"""
    rand_t = torch.rand(3, 4)  #3x4 случайные числа от 0 до 1
    zeros_t = torch.zeros(2, 3, 4)  #2x3x4 из нулей
    ones_t = torch.ones(5, 5)  #5x5 из единиц
    range_t = torch.arange(16).reshape(4, 4)  #числа 0..15 и превращаем в таблицу 4x4
    return rand_t, zeros_t, ones_t, range_t


def tensor_operations():
    """Операции из пункта 1.2: A (3x4) и B (4x3)"""
    A = torch.rand(3, 4)
    B = torch.rand(4, 3)

    #проверка: для матричного умножения число столбцов A должно равняться числу строк B
    if A.shape[1] != B.shape[0]:
        raise ValueError("размеры не подходят")

    A_t = A.T  #транспонирование A(смена строк и столбцов), станет 4x3
    matmul = A @ B  #матричное умножение 3x4 на 4x3 даёт 3x3
    elementwise = A * B.T  #поэлементное умножение: B.T имеет размер 3x4, как и A
    total = A.sum()  #сумма всех элементов A
    return A, B, A_t, matmul, elementwise, total


def indexing_demo():
    """Индексация и срезы из пункта 1.3 на тензоре 5x5x5"""
    x = torch.arange(125).reshape(5, 5, 5)  #числа от 0 до 124 в форме 5x5x5

    first_row = x[0, 0, :]  #первая строка (первой матрицы), 5 элементов
    last_column = x[0, :, -1]  #последний столбец (первой матрицы), 5 элементов
    center = x[2, 1:3, 1:3]  #подматрица 2x2 из центральной матрицы (индекс 2), строки и столбцы 1 и 2
    even = x[::2, ::2, ::2]  #элементы с чётными индексами 0, 2, 4 по каждой оси, получится 3x3x3
    return x, first_row, last_column, center, even


def reshape_to(t, shape):
    """Меняет форму тензора t на shape, но сначала проверяет, что элементов хватает"""
    if t.numel() != math.prod(shape):  #numel - сколько всего элементов prod - произведение размеров новой формы
        raise ValueError(f"Нельзя преобразовать {t.numel()} элементов в форму {shape}")
    return t.reshape(shape)


def reshape_demo():
    """Пункт 1.4: из 24 элементов делаем разные формы"""
    t = torch.arange(24)  #числа 0..23
    shapes = [(2, 12), (3, 8), (4, 6), (2, 3, 4), (2, 2, 2, 3)]
    return [reshape_to(t, s) for s in shapes]  #для каждой формы вызываем reshape_to


def run_tests():
    """Простые проверки (если что-то не так, будет ошибка AssertionError)"""
    rand_t, zeros_t, ones_t, range_t = create_tensors()
    assert rand_t.shape == (3, 4) and rand_t.min() >= 0 and rand_t.max() <= 1
    assert zeros_t.shape == (2, 3, 4) and zeros_t.sum() == 0
    assert ones_t.shape == (5, 5) and ones_t.sum() == 25
    assert range_t.shape == (4, 4) and range_t[3, 3] == 15

    A, B, A_t, matmul, elementwise, total = tensor_operations()
    assert A_t.shape == (4, 3)
    assert matmul.shape == (3, 3)
    assert elementwise.shape == (3, 4)
    assert torch.isclose(total, A.flatten().sum())  #сумма совпадает с суммой по сплюснутому тензору

    x, first_row, last_column, center, even = indexing_demo()
    assert first_row.tolist() == [0, 1, 2, 3, 4]
    assert last_column.tolist() == [4, 9, 14, 19, 24]
    assert center.tolist() == [[56, 57], [61, 62]]
    assert even.shape == (3, 3, 3)

    shapes = [t.shape for t in reshape_demo()]
    assert shapes == [(2, 12), (3, 8), (4, 6), (2, 3, 4), (2, 2, 2, 3)]
    try:
        reshape_to(torch.arange(24), (5, 5))  #такую форму сделать нельзя
        assert False, "должна была быть ошибка"
    except ValueError:
        pass  #ошибка появилась
    print("Все проверки пройдены")


if __name__ == "__main__":
    rand_t, zeros_t, ones_t, range_t = create_tensors()
    print("1.1 Создание тензоров")
    print("rand 3x4:\n", rand_t)
    print("zeros 2x3x4, форма:", zeros_t.shape)
    print("ones 5x5:\n", ones_t)
    print("arange 0..15 в форме 4x4:\n", range_t)

    A, B, A_t, matmul, elementwise, total = tensor_operations()
    print("\n1.2 Операции с тензорами")
    print("A:\n", A)
    print("A транспонированная, форма:", A_t.shape)
    print("A @ B:\n", matmul)
    print("A * B.T:\n", elementwise)
    print("сумма элементов A:", total.item())

    x, first_row, last_column, center, even = indexing_demo()
    print("\n1.3 Индексация и срезы")
    print("первая строка:", first_row)
    print("последний столбец:", last_column)
    print("центральная подматрица 2x2:\n", center)
    print("чётные индексы, форма:", even.shape)

    print("\n1.4 Работа с формами")
    for t in reshape_demo():
        print(tuple(t.shape))

    print()
    run_tests()
