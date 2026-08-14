from packvium import Container, Dimensions, Item, Packer, PackingConfig

result = Packer(PackingConfig.balanced()).pack(
    [Item.create("book", Dimensions.mm("210", "140", "30"), "450 g", quantity=4), Item.create("mug", Dimensions.inches("4", "4", "5"), "12 oz", quantity=2, keep_upright=True)],
    [Container.create("box-m", Dimensions.mm("400", "300", "250"), max_payload="20 kg", cost_minor=180), Container.create("box-l", Dimensions.mm("500", "400", "350"), max_payload="30 kg", cost_minor=250)],
)
print(result.to_dict())
