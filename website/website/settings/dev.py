from base import *

DEV_DIR = os.path.join(BASE_DIR, 'temp')
MEDIA_ROOT = DEV_DIR
MEDIA_URL = ''

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '1^#W5u1O^$O3)m{!4i[A32KA0KMn+[]Jv#1)xwt!/Q[8Nh&,vA!KL18^.ObQe|rz'


# Database
# https://docs.djangoproject.com/en/1.6/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': os.path.join(DEV_DIR, 'dev.db'),
    }
}
