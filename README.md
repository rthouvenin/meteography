Meteography
===========

Maybe you have seen your grandfather look at the horizon and tell you what the weather will be the day after. That is what Meteography is: using a stream of photos from a fixed webcam to predict what the sky will be like in the near future.
It does not forecast the weather in terms of temperature, wind, rain, etc. but rather generate an image picturing what the webcam should capture later.

Meteography is written in Python using (among others) Pytables and Scikit-learn. The web part is done with Django, so it can be deployed on any web server that supports serving WSGI applications.

----------

### See
There is a public deployment (used for tests) where you can see the progress of the project and what it looks like: http://ciel.fastu.eu

### Install (for dev)
The project is still in early development phase. But if you want to install it anyway to play with it, or want to contribute, you can.
To do so, clone the source and optionally create a virtual environment if you don't want to affect your system packages (you can also use the `--user` option of pip).

Then, from the root directory of the project:

    $ python setup.py sdist
    $ pip install -f dist meteography

This will automatically install the required dependencies, but note that some of them (matplotlib and pytables) may be more easily installed first using your system distribution.

### Setup (for dev)
Once the project is installed, before using it for the first time you need to create the database and a superuser account.
From the root directory of the project:

    $ cd website/
    $ mkdir data cams
    $ python manage.py migrate
    $ create super user: 'python manage.py createsuperuser'

### Use
You can now start the server (still from website directory):

     $ python manage.py runserver

 Now to start feeding images to the application:

 - Go to the admin section: http://127.0.0.1:8000/admin
 - Log in with the superuser created earlier
 - Add a webcam
 - Add a prediction params (to be documented)
 - Upload pictures with the API (to be documented)

### More information
Please contact the author.
