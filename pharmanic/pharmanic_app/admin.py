from django.contrib import admin
from django.utils.html import format_html
from django.urls import reverse
from .models import (
Customer,
Product,
Cart,
OrderPlaced,
ViewCount,
CartCount,
Rating,
FileModel
)

@admin.register(Customer)
class CustomerModelAdmin(admin.ModelAdmin):
	list_display = ['id', 'user', 'name', 'locality', 'city', 'zipcode', 'state']

@admin.register(Product)
class ProductModelAdmin(admin.ModelAdmin):
	list_display = ['id', 'title', 'searchname','selling_price', 'discounted_price', 'description', 'brand', 'category', 'product_image']

@admin.register(Cart)
class CartModelAdmin(admin.ModelAdmin):
	list_display = ['id', 'user', 'product', 'quantity']

@admin.register(OrderPlaced)
class OrderPlacedModelAdmin(admin.ModelAdmin):
	list_display = ['id', 'user', 'customer','customer_info', 'product','product_info', 'quantity', 'ordered_date', 'status']

	def customer_info(self, obj):
		link = reverse("admin:pharmanic_app_customer_change", args=[obj.customer.pk])
		return format_html('<a href="{}">{}</a>', link, obj.customer.name)

	def product_info(self, obj):
		link = reverse("admin:pharmanic_app_product_change", args=[obj.product.pk])
		return format_html('<a href="{}">{}</a>', link, obj.product.title)

@admin.register(ViewCount)
class ViewCountModelAdmin(admin.ModelAdmin):
	list_display = ['id', 'pname', 'count']

@admin.register(CartCount)
class CartCountModelAdmin(admin.ModelAdmin):
	list_display = ['id', 'pname', 'count']
	
@admin.register(Rating)
class RatingModelAdmin(admin.ModelAdmin):
	list_display = ['id', 'userid', 'pname', 'p_category', 'rating']

@admin.register(FileModel)
class FileModelAdmin(admin.ModelAdmin):
	list_display = ['id', 'file']