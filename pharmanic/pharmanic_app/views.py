
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
from sklearn.decomposition import TruncatedSVD
from typing import Dict, Text
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import seaborn as sns
import matplotlib.pyplot as plt

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
    api_key = "AIzaSyCmVndYRVeskgFSBI7v1YXlFc218txz47k"
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
            # print(img)
            if(len(FileModel.objects.filter(file=img['file']))) != 0:
                c = FileModel.objects.get(file=img['file'])
                c.file.delete()
                c.delete()
                
            f.save()
            img_path = './media/' + str(img['file'])
            # print(img_path)
            res = visionapi(img_path)
            text = []
            for i in res:
                if len(i) > 2 and not i.isdigit():
                    text.append(i)
            print(text)
            search = Product.objects.none()
            # print(search)
            for i in text:
                search |= Product.objects.filter(Q(searchname__icontains=i) & Q(category='TC'))
            # for i in search:
            #     print(i.title)

            totalitem = 0
            if request.user.is_authenticated:
                totalitem = len(Cart.objects.filter(user=request.user))
            return render(request, 'app/prescription_medicines.html', {'prescription_medicines': search,'totalitem': totalitem})
            # fm = FileForm()
            # return render(request, "app/orderwithprescription.html", {'fm':fm, 'msg':text})
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

        recomm_products_names = collaborativeFiltering()
        print(recomm_products_names)
        recomm_products = Product.objects.filter(title__in=recomm_products_names)
        print(recomm_products)
        # print(fetch_data())
        if request.user.is_authenticated:
            totalitem = len(Cart.objects.filter(user=request.user))
        return render(request, 'app/home.html', {'recomm_products': recomm_products, 'tablets_capsules': tablets_capsules, 'supplements': supplements, 'general': general, 'herbs': herbs, 'health_drinks': health_drinks, 'ayurvedic': ayurvedic, 'covid_essentials': covid_essentials, 'personal_care': personal_care, 'healthcare': healthcare,'totalitem': totalitem})

def fetch_data():
    # products = pd.DataFrame(list(Product.objects.all().values()))
    carts = pd.DataFrame(list(CartCount.objects.all().values()))
    views = pd.DataFrame(list(ViewCount.objects.all().values()))
    # ratings = pd.DataFrame(list(Rating.objects.all().values()))
    # ratings['userid'] = ratings['userid'].apply(str)
    # ratings['rating'] = ratings['rating'].astype(float)
    data = preprocess(ratings,products,userid)
    popular_cart_products, popular_view_products = cart_view_prediction(carts, views)
    recommended_products_ratings = []
    for i in range(0, data.shape[0]):
        recommended_products_ratings.append(data[i].numpy().decode())

    recommended_products_util = common_recommended_products(recommended_products_ratings, popular_cart_products)
    recommended_products = common_recommended_products(recommended_products_util, popular_view_products)

    recommended_products = recommended_products + recommended_products_util
    recommended_products = recommended_products + recommended_products_ratings

    if len(recommended_products) > 5:
        extra_elements = len(recommended_products) - 5
        del recommended_products[-extra_elements]
    
    return recommended_products

def common_recommended_products(list1, list2):
    return list(set(list1) & set(list2))


def cart_view_prediction(carts, views):
    popular_cart_products = carts.sort_values(by=['count'], ascending=False)
    popular_view_products = views.sort_values(by=['count'], ascending=False)
    return popular_cart_products, popular_view_products

# For new users
def collaborativeFiltering():
    ratings = pd.DataFrame(Rating.objects.all().values())
    carts = pd.DataFrame(CartCount.objects.all().values())
    popular_cart_products = carts.sort_values(by=['count'], ascending=False)
    views = pd.DataFrame(ViewCount.objects.all().values())
    popular_view_products = views.sort_values(by=['count'], ascending=False)
    
    # print(ratings)
    ratings_with_id = ratings[['userid', 'pname', 'rating']]
    # Need to find product_id using pname
    pname = ratings['pname']
    # print(list(pname))
    p_id = []
    for i in pname:
        id = Product.objects.get(title = i).id
        p_id.append(id)
    # print(p_id)
    ratings_with_id['product_id'] = p_id
    ratings_utility_matrix = ratings_with_id.pivot_table(values='rating', index='userid', columns='product_id', fill_value=0)
    X = ratings_utility_matrix.T
    print(ratings.userid)
    svd = TruncatedSVD(n_components=len(np.unique(ratings.userid)))
    decomposed_matrix = svd.fit_transform(X)
    correlation_matrix = np.corrcoef(decomposed_matrix)
    for product_ in p_id:
        product_titles = list(X.index)
        product_id = product_titles.index(product_)
        correlation_product_id = correlation_matrix[product_id]
        recommend = list(X.index[correlation_product_id > 0.90])
        recommend.remove(product_)
        
    
    recommended_prod = []
    for i in recommend:
        recommended_prod.append(Product.objects.get(id=i).title)


    # print(recommend)
    print("*****************************************************************************************************")
    print(decomposed_matrix)
    print("*****************************************************************************************************")
    print(X)
    
    corr = sns.heatmap(decomposed_matrix)
    plt.savefig("corr.png")


    cart_recomm = popular_cart_products.pname.to_list()
    # print(cart_recomm)

    view_recomm = popular_view_products.pname.to_list()
    # print(view_recomm)

    recommended_products = recommended_prod[:4] 

    k = 0
    for i in cart_recomm:
        if k == 3:
            break
        if i not in recommended_products:
            recommended_products.append(i)
            k += 1

    n = 0
    for i in view_recomm:
        if n == 3:
            break
        if i not in recommended_products:
            recommended_products.append(i)
            n += 1

    return recommended_products










# def product_detail(request):
# return render(request, 'app/productdetail.html')

class ProductDetailView(View):
    def get(self, request, pk):
        totalitem = 0
        product = Product.objects.get(pk=pk)
        pname = product.title
        ordered = False
        item_already_in_cart = False
        rating = 0
        if len(ViewCount.objects.filter(pname=pname))!=0:
                count=ViewCount.objects.filter(pname=pname)[0].count+1
                ViewCount.objects.filter(pname=pname).update(count=count)
        else:
            obj=ViewCount(pname=pname,count=1)
            obj.save()

        if request.user.is_authenticated:   
            id = request.user.id
            obj1 = Rating.objects.filter(userid=id, pname=pname)

            #If the user has changed the rating
            if len(obj1)!=0:
                obj2 = Rating.objects.get(userid=id,pname=pname)
                rating = obj2.rating
            totalitem = len(Cart.objects.filter(user=request.user))
            item_already_in_cart = Cart.objects.filter(Q(product=product.id) & Q(user=request.user)).exists()
            ordered = OrderPlaced.objects.filter(Q(product=product.id) & Q(user=request.user)).exists()
            
        return render(request, 'app/productdetail.html',
        {'product': product, 'ordered': ordered,'rating': int(rating),'item_already_in_cart': item_already_in_cart,'totalitem': totalitem})


    def post(self, request, pk):
        if request.method=='POST':
            product = Product.objects.get(pk=pk)
            pname = product.title
            pcategory = product.category
            rating=request.POST.get('rate')
            # print(rating)
            # print('8888888888888888888888888888')
            
            if request.user.is_authenticated:
                id = request.user.id
                obj1 = Rating.objects.filter(userid=id, pname=pname)

                #If the user has changed the rating
                if len(obj1)!=0:
                    Rating.objects.filter(userid=id,pname=pname,p_category=pcategory).update(rating=rating)
                #User is rating for first time
                else:   
                    obj1=Rating(userid=id,pname=pname,rating=rating,p_category=pcategory)
                    obj1.save()

            return render(request, 'app/product_rating.html')





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
    dollaramount = 0.0
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
    dollaramount = totalamount/82.49
    return render(request, 'app/checkout.html', {'add':add, 'totalamount':totalamount, 'dollaramount': dollaramount, 'cart_items': cart_items,'totalitem': totalitem})

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
