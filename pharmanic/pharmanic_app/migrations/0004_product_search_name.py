# Generated by Django 2.2.28 on 2023-03-04 07:24

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('pharmanic_app', '0003_filemodel'),
    ]

    operations = [
        migrations.AddField(
            model_name='product',
            name='search_name',
            field=models.CharField(blank=True, max_length=100),
        ),
    ]