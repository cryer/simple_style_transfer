# simple_style_transfer
simple style transfer with a very clear code

## Note

I have created a repo about style transfer several months ago.It is implemented referring to official tutorial.
And its code may be a little hard to read and understand.However,this code is very simple and clear,and has a 
lot of tricks.You may understand them and use them in your own porjects.

## Usage

```
git clone https://github.com/cryer/simple_style_transfer.git
cd simple_style_transfer
pip install -r requirements.txt
python train.py train
```
Add additional configs:
```
    --content = './image/gakki.jpeg'
    --style = './image/in1.png'
    --max_size = 400
    --total_step = 5000
    --log_step = 10
    --sample_step = 1000
    --style_weight = 100
    --lr = 0.003
    --use_gpu = True
```
Pay attention to images' path.
## Results

Use my idol gakki to do a test.

本当に　かわいいねｗｗ　女はみずだ！

| content image|style image |transfer image|
|:----------:|:----------:|:----------:|
|<img src="https://github.com/cryer/simple_style_transfer/raw/master/image/gakki.jpeg" width = "1000" height = "300"/>|<img src="https://github.com/cryer/simple_style_transfer/raw/master/image/in1.png" width = "1000" height = "300"/>|<img src="https://github.com/cryer/simple_style_transfer/raw/master/image/saved_picture23.png" width = "1000" height = "300"/>|

## Difference

Compare to my previous style transfer repo or official implementation,this repo's computation of content loss is different.
Using five layers' content loss togeter instead of only one layer's content loss in paper and official implementation.So
it may take longer time,but a little bit improvement a think,because if it improves a lot , paper may likely use this kind of loss.
You can change loss computation part to do a test.

## Reference

[Yunjey](https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/neural_style_transfer),Yunjey's code is always
very clear,you may learn a lot from him.
