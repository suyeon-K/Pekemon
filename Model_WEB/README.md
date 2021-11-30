# Pekemon_Model_Web

Using: FLASK and AWS

To install requirements.txt

```
pip install -r requirements.txt
```

Check rest api code at local server

```
import requests
resp = requests.post("http://localhost:5000/predict{}".format(choose),
                   	    files={"file": open('input.jpg','rb')})
# Save result picture
save_path = "result{}.jpg".format(choose)
photo = open(save_path, 'wb')
photo.write(resp.content)
photo.close()
```

Check rest api code at AWS server(Now server is closed.)

```
import requests
resp = requests.post("http://18.219.72.141:5000/predict{}".format(choose),
                   	    files={"file": open('input.jpg','rb')})
# Save result picture
save_path = "result{}.jpg".format(choose)
photo = open(save_path, 'wb')
photo.write(resp.content)
photo.close()
```

***

Reference model: [Jihye Back](https://github.com/happy-jihye/Cartoon-StyleGAN)
