CREATE OR REPLACE PROCEDURE refresh_order_totals()
LANGUAGE SQL
AS $$
    UPDATE orders SET total = subtotal + tax_amount;
$$;

CREATE FUNCTION order_total(order_id integer) RETURNS numeric
LANGUAGE SQL AS $$ SELECT total FROM orders WHERE id = order_id $$;
