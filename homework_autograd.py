"""Задание 2. Автоматическое дифференцирование"""
import torch


def task_2_1():
    """f(x, y, z) = x^2 + y^2 + z^2 + 2*x*y*z, градиенты по x, y, z"""
    x = torch.tensor(1.0, requires_grad=True)  #requires_grad=True - следить за операциями, чтобы потом посчитать градиент
    y = torch.tensor(2.0, requires_grad=True)
    z = torch.tensor(3.0, requires_grad=True)

    f = x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z  # сама функция
    f.backward()  #считаем градиенты по всем переменным

    #аналитические производные (считаем руками по формулам):
    #df/dx = 2x + 2yz, df/dy = 2y + 2xz, df/dz = 2z + 2xy
    dx = 2 * x.detach() + 2 * y.detach() * z.detach()  #detach() - берём число без слежения за градиентом
    dy = 2 * y.detach() + 2 * x.detach() * z.detach()
    dz = 2 * z.detach() + 2 * x.detach() * y.detach()

    print("f =", f.item())
    print("torch:       ", x.grad.item(), y.grad.item(), z.grad.item())
    print("аналитически:", dx.item(), dy.item(), dz.item())
    assert torch.allclose(x.grad, dx) and torch.allclose(y.grad, dy) and torch.allclose(z.grad, dz)


def mse_loss(y_pred, y_true):
    """MSE = (1/n) * сумма (y_pred - y_true)^2"""
    if y_pred.shape != y_true.shape:  #проверка: размеры должны совпадать
        raise ValueError("y_pred и y_true должны быть одного размера")
    return ((y_pred - y_true) ** 2).mean()  #квадраты ошибок и их среднее


def task_2_2():
    """Линейная функция y_pred = w * x + b, градиенты MSE по w и b"""
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])  #входы (градиент по ним не нужен)
    y_true = torch.tensor([3.0, 5.0, 7.0, 9.0])  #правильные ответы (здесь y = 2x + 1)
    w = torch.tensor(0.5, requires_grad=True)  #начальные значения параметров
    b = torch.tensor(0.0, requires_grad=True)

    y_pred = w * x + b  #прогноз линейной функции
    loss = mse_loss(y_pred, y_true)
    loss.backward()  #градиенты loss по w и b

    # аналитически: dMSE/dw = (2/n) * сумма (err * x), dMSE/db = (2/n) * сумма err, где err = y_pred - y_true
    n = x.numel()
    err = y_pred.detach() - y_true
    dw = 2 / n * (err * x).sum()
    db = 2 / n * err.sum()

    print("MSE =", loss.item())
    print("torch:       dw =", w.grad.item(), " db =", b.grad.item())
    print("аналитически: dw =", dw.item(), " db =", db.item())
    assert torch.allclose(w.grad, dw) and torch.allclose(b.grad, db)


def task_2_3():
    """Цепное правило: f(x) = sin(x^2 + 1), df/dx = cos(x^2 + 1) * 2x."""
    x = torch.tensor([0.5, 1.0, 1.5, 2.0], requires_grad=True)
    f = torch.sin(x ** 2 + 1)  #составная функция: сначала x^2 + 1, потом sin
    total = f.sum()  #backward работает со скаляром, поэтому суммируем (производная каждого элемента при этом не меняется)

    grad_auto = torch.autograd.grad(total, x, retain_graph=True)[0]  #градиент через torch.autograd.grad
    total.backward()  #градиент через backward (кладётся в x.grad)

    xd = x.detach()
    analytic = torch.cos(xd ** 2 + 1) * 2 * xd  #производная по цепному правилу

    print("backward:           ", x.grad)
    print("torch.autograd.grad:", grad_auto)
    print("аналитически:       ", analytic)
    assert torch.allclose(x.grad, analytic, atol=1e-6)
    assert torch.allclose(grad_auto, analytic, atol=1e-6)


if __name__ == "__main__":
    print("2.1 Простые вычисления с градиентами")
    task_2_1()
    print("\n2.2 Градиент функции потерь (MSE)")
    task_2_2()
    print("\n2.3 Цепное правило")
    task_2_3()
    print("\nВсе проверки пройдены")
