package benchmark.orders;

public class OrderService {
    public Money calculateTotal(Order order) {
        return taxCalculator.apply(order.subtotal());
    }

    private Money persistTotal(Order order, Money total) {
        return repository.save(order.id(), total);
    }
}
