ORDERS = {"1042": "shipped", "1043": "processing", "1044": "delivered"}


def get_order_status(order_id: str) -> str:
    return ORDERS.get(order_id, "not found")


TOOLS = {
    "get_order_status": {
        "fn": get_order_status,
        "description": "Look up the status of a customer order by its numeric id.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    }
}
