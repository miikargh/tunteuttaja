# Tunteuttaja
A training script, an API, and an app for a transformer that takes in Finnish text and spits
out emojis.

## Quick start guide
If you have docker installed on your computer the fastest way to test the API is
to run the following command (the docker file is quite big because it includes
the transformer model):

```sh
$ docker run -itd -p 9090:9090 miikargh/tunteuttaja:0.1.0
```

The above command will pull and run the image and run the API on port 9090. Just
head to http://localhost:9090/app/index.html to test it out! Alternatively you
can check http://localhost:9090/docs out for the API documentation.
