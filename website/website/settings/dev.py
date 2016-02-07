from base import *


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/1.8/howto/deployment/checklist/

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True
TEMPLATE_DEBUG = True
ALLOWED_HOSTS = []


# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '1^#W5u1O^$O3)m{!4i[A32KA0KMn+[]Jv#1)xwt!/Q[8Nh&,vA!KL18^.ObQe|rz'

INSTALLED_APPS = INSTALLED_APPS + ('django_extensions',)

# Database
# https://docs.djangoproject.com/en/1.8/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(DATA_DIR, 'dev.db'),
    }
}
