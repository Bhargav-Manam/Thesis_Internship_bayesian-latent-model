from data.synthetic import Generator
n = 5
d = [1, 2, 3]
q = 2
min_len = 1
max_len = 4

data_generator = Generator(n, d, q, min_len, max_len)
y_n = data_generator.y_n