from feast import Entity
from feast.value_type import ValueType

customer = Entity(
    name="customer",
    join_keys=["Customer_ID"],
    value_type=ValueType.INT64,
    description="A single customer of the e-commerce platform",
)
