def finalize_invoice(db, invoice_id):
    rows = db.fetch("SELECT * FROM invoice_runs WHERE invoice_id = %s", (invoice_id,))
    if not rows:
        raise RuntimeError("invoice_rows missing")
    return rows[0]
