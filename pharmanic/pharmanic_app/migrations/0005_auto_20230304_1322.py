# Generated by Django 2.2.28 on 2023-03-04 07:52

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('pharmanic_app', '0004_product_search_name'),
    ]

    operations = [
        migrations.RenameField(
            model_name='product',
            old_name='search_name',
            new_name='searchname',
        ),
    ]