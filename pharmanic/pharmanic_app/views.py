
from __future__ import print_function
from django.shortcuts import redirect, render
from django.views import View
from .models import Customer, Product, Cart, OrderPlaced, ViewCount, CartCount, Rating, FileModel
from .forms import CustomerProfileForm, CustomerRegistrationForm, FileForm
from django.contrib import messages
from django.db.models import Q
from django.http import JsonResponse
from django.contrib.auth.decorators import login_required
from django.utils.decorators import method_decorator

from base64 import b64encode
from os import makedirs, remove
from os.path import join, basename
from sys import argv
import json
import requests
import glob
from unidecode import unidecode
from django.conf import settings

# def home(request):
# return render(request, 'app/home.html')

ENDPOINT_URL = 'https://vision.googleapis.com/v1/images:annotate'
RESULTS_DIR = 'jsons'

def make_image_data_list(image_filenames):
    """
    image_filenames is a list of filename strings
    Returns a list of dicts formatted as the Vision API
        needs them to be
    """
    img_requests = []
    with open(image_filenames, 'rb') as f:
        ctxt = b64encode(f.read()).decode()
        img_requests.append({
                'image': {'content': ctxt},
                'features': [{
                    'type': 'TEXT_DETECTION',
                    'maxResults': 1
                }]
        })
    return img_requests

def make_image_data(image_filenames):
    """Returns the image data lists as bytes"""
    imgdict = make_image_data_list(image_filenames)
    return json.dumps({"requests": imgdict }).encode()


def request_ocr(api_key, image_filenames):
    response = requests.post(ENDPOINT_URL, data=make_image_data(image_filenames), params={'key': api_key}, headers={'Content-Type': 'application/json'})
    return response

def remove_non_ascii(text):
    return unidecode(str(text, encoding = "utf-8"))

def visionapi(img):    
    api_key = "AIzaSyBRB_d6Ahac8EfQPklTo6uKPCFDICrVDdg"
    image_filenames = img
    response = request_ocr(api_key, image_filenames)
    # print(type(response))

    # print(type(response.text))
    # print(response.text)

    entities = []
    for i in range(1,len(response.json()['responses'][0]['textAnnotations'])):
        entities.append(remove_non_ascii(response.json()['responses'][0]['textAnnotations'][i]['description'].encode("utf-8")))
    return (entities)


def orderwithprescription(request):
    if request.method == "POST":
        f = FileForm(request.POST, request.FILES)
        if f.is_valid():
            img = request.FILES
            print(img)
            if(len(FileModel.objects.filter(file=img['file']))) != 0:
                c = FileModel.objects.get(file=img['file'])
                c.file.delete()
                c.delete()
                
            f.save()
            img_path = './media/' + str(img['file'])
            print(img_path)
            text = visionapi(img_path)

            fm = FileForm()
            return render(request, "app/orderwithprescription.html", {'fm':fm, 'msg':text})
        else:
            return render(request, "app/orderwithprescription.html", {'fm':f, 'msg':'Check Errors!'})
    else:
        fm = FileForm()
        return render(request, "app/orderwithprescription.html", {'fm':fm})
    # return render(request, 'app/orderwithprescription.html')

class ProductView(View):
    def get(self, request):
        totalitem = 0
        tablets_capsules = Product.objects.filter(category='TC')
        supplements = Product.objects.filter(category='S')
        general = tablets_capsules | supplements
        herbs = Product.objects.filter(category='H')
        health_drinks = Product.objects.filter(category='HD')
        ayurvedic = herbs | health_drinks
        covid_essentials = Product.objects.filter(category='CE')
        personal_care = Product.objects.filter(category='PC')
        healthcare = covid_essentials | personal_care
        if request.user.is_authenticated:
            totalitem = len(Cart.objects.filter(user=request.user))
        return render(request, 'app/home.html', {'tablets_capsules': tablets_capsules, 'supplements': supplements, 'general': general, 'herbs': herbs, 'health_drinks': health_drinks, 'ayurvedic': ayurvedic, 'covid_essentials': covid_essentials, 'personal_care': personal_care, 'healthcare': healthcare,'totalitem': totalitem})


# def product_detail(request):
# return render(request, 'app/productdetail.html')

class ProductDetailView(View):
    def get(self, request, pk):
        totalitem = 0
        product = Product.objects.get(pk=pk)
        pname = product.title
        #print(pname)
        item_already_in_cart = False
        if request.user.is_authenticated:
            if len(ViewCount.objects.filter(pname=pname))!=0:
                count=ViewCount.objects.filter(pname=pname)[0].count+1
                ViewCount.objects.filter(pname=pname).update(count=count)
            else:
                obj=ViewCount(pname=pname,count=1)
                obj.save()
                
            totalitem = len(Cart.objects.filter(user=request.user))
            item_already_in_cart = Cart.objects.filter(Q(product=product.id) & Q(user=request.user)).exists()

        return render(request, 'app/productdetail.html',
        {'product': product, 'item_already_in_cart': item_already_in_cart,'totalitem': totalitem})

@login_required
def add_to_cart(request): 
    user = request.user
    product_id = request.GET.get('prod_id')
    product = Product.objects.get(id=product_id)
    pname = product.title
    if request.user.is_authenticated:
        if len(CartCount.objects.filter(pname=pname))!=0:
            count=CartCount.objects.filter(pname=pname)[0].count+1
            CartCount.objects.filter(pname=pname).update(count=count)
        else:
           obj=CartCount(pname=pname,count=1)
           obj.save()
    Cart(user=user, product=product).save()
    return redirect('/cart')

@login_required
def show_cart(request):
    totalitem = 0
    if request.user.is_authenticated:
        user = request.user
        cart = Cart.objects.filter(user=user)
        totalitem = len(Cart.objects.filter(user=request.user))
        # print(cart)
        if totalitem == 0:
            return render(request, 'app/emptycart.html')

        amount = 0.0
        shipping_amount = 70.0
        total_amount = 0.0
        cart_product = [p for p in Cart.objects.all() if p.user == user]
        # print(cart_product)
        if cart_product:
            for p in cart_product:
                tempamount = (p.quantity * p.product.discounted_price)
                amount += tempamount
                totalamount = amount+shipping_amount
            return render(request, 'app/addtocart.html', {'totalamount': totalamount, 'amount': amount, 'carts': cart,'totalitem': totalitem})
        else:
            return render(request, 'app/emptycart.html')

def plus_cart(request):
    if request.method == 'GET':
        prod_id = request.GET['prod_id']
        c = Cart.objects.get(Q(product=prod_id) & Q(user=request.user))
        c.quantity+=1
        c.save()
        amount = 0.0
        shipping_amount = 70.0
        cart_product = [p for p in Cart.objects.all() if p.user == request.user]
        for p in cart_product:
            tempamount = (p.quantity * p.product.discounted_price)
            amount += tempamount

        data = {
         'quantity': c.quantity,
         'amount': amount,
         'totalamount': amount+shipping_amount
        }
        return JsonResponse(data)

def minus_cart(request):
    if request.method == 'GET':
        prod_id = request.GET['prod_id']
        c = Cart.objects.get(Q(product=prod_id) & Q(user=request.user))
        c.quantity-=1
        if c.quantity <= 0:
            remove_cart(int(prod_id))
	 
        c.save()
        amount = 0.0
        shipping_amount = 70.0
        cart_product = [p for p in Cart.objects.all() if p.user == request.user]
        for p in cart_product:
            tempamount = (p.quantity * p.product.discounted_price)
            amount += tempamount

        data = {
         'quantity': c.quantity,
         'amount': amount,
         'totalamount': amount + shipping_amount
        }
        return JsonResponse(data)


def remove_cart(request):
    if request.method == 'GET':
        prod_id = request.GET['prod_id']
        c = Cart.objects.get(Q(product=prod_id) & Q(user=request.user))
        c.delete()
        amount = 0.0
        shipping_amount = 70.0
        cart_product = [p for p in Cart.objects.all() if p.user == request.user]
        for p in cart_product:
            tempamount = (p.quantity * p.product.discounted_price)
            amount += tempamount

        data = {
         'amount': amount,
         'totalamount': amount + shipping_amount
        }
        return JsonResponse(data)


def buy_now(request):
    return render(request, 'app/buynow.html')


def profile(request):
    return render(request, 'app/profile.html')

@login_required
def address(request):
    totalitem = 0
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
    add = Customer.objects.filter(user=request.user)
    return render(request, 'app/address.html', {'add': add, 'totalitem': totalitem})

@login_required
def orders(request):
    op = OrderPlaced.objects.filter(user=request.user)
    totalitem = 0
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
    return render(request, 'app/orders.html', {'order_placed':op,'totalitem': totalitem})


def tablets_capsules(request):
    tablets_capsules = Product.objects.filter(category='TC')
    totalitem = 0
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
    return render(request, 'app/tablets_capsules.html', {'tablets_capsules': tablets_capsules,'totalitem': totalitem})


def supplements(request):
    supplements = Product.objects.filter(category='S')
    totalitem = 0
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
    return render(request, 'app/supplements.html', {'supplements': supplements,'totalitem': totalitem})


def herbs(request):
    herbs = Product.objects.filter(category='H')
    totalitem = 0
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
    return render(request, 'app/herbs.html', {'herbs': herbs,'totalitem': totalitem})


def health_drinks(request):
    health_drinks = Product.objects.filter(category='HD')
    totalitem = 0
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
    return render(request, 'app/health_drinks.html', {'health_drinks': health_drinks,'totalitem': totalitem})


def covid_essentials(request):
    covid_essentials = Product.objects.filter(category='CE')
    totalitem = 0
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
    return render(request, 'app/covid_essentials.html', {'covid_essentials': covid_essentials,'totalitem': totalitem})


def personal_care(request):
    personal_care = Product.objects.filter(category='PC')
    totalitem = 0
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
    return render(request, 'app/personal_care.html', {'personal_care': personal_care,'totalitem': totalitem})


class CustomerRegistrationView(View):
    def get(self, request):
        totalitem = 0
        form = CustomerRegistrationForm()
        if request.user.is_authenticated:
            totalitem = len(Cart.objects.filter(user=request.user))
        return render(request, 'app/customerregistration.html', {'form': form,'totalitem': totalitem})

    def post(self, request):
        form = CustomerRegistrationForm(request.POST)
        if form.is_valid():
            messages.success(request,'Congratulations!! Registered Successfully')
            form.save()
        return render(request, 'app/customerregistration.html', {'form': form})

@login_required
def checkout(request):
    user = request.user
    add = Customer.objects.filter(user=user)
    cart_items = Cart.objects.filter(user=user)
    amount = 0.0
    shipping_amount = 70.0
    totalamount = 0.0
    cart_product = [p for p in Cart.objects.all() if p.user == request.user]
    if cart_product:
        for p in cart_product:
            tempamount = (p.quantity * p.product.discounted_price)
            amount += tempamount
        print(amount)
        totalamount = amount + shipping_amount
    else:
        return redirect(show_cart)
    if request.user.is_authenticated:
        totalitem = len(Cart.objects.filter(user=request.user))
    return render(request, 'app/checkout.html', {'add':add, 'totalamount':totalamount, 'cart_items': cart_items,'totalitem': totalitem})

@login_required
def payment_done(request):
    user = request.user
    custid = request.GET.get('custid')
    customer = Customer.objects.get(id=custid)
    cart = Cart.objects.filter(user=user)
    for c in cart:
        OrderPlaced(user=user, customer=customer, product=c.product, quantity=c.quantity).save()
        c.delete()
    return redirect("orders")


@method_decorator(login_required, name='dispatch')
class ProfileView(View):
    def get(self, request):
        form = CustomerProfileForm
        totalitem = 0
        if request.user.is_authenticated:
            totalitem = len(Cart.objects.filter(user=request.user))
        return render(request, 'app/profile.html', {'form': form,'totalitem': totalitem})

    def post(self, request):
        form = CustomerProfileForm(request.POST)
        if form.is_valid():
            usr = request.user
            name = form.cleaned_data['name']
            locality = form.cleaned_data['locality']
            city = form.cleaned_data['city']
            state = form.cleaned_data['state']
            zipcode = form.cleaned_data['zipcode']
            reg = Customer(user=usr, name=name, locality=locality,city=city,state=state,zipcode=zipcode)
            reg.save()
            messages.success(request,'Congratulations !! profile updated successfully')
        return render(request, 'app/profile.html', {'form': form, 'active': 'btn-primary'})
