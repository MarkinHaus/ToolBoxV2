class Code:
    @staticmethod
    def decode_code(data):
        # letters = string.ascii_letters + string.digits + string.punctuation
        # decode_str = ''
        # data_n = data.split('#')
        # data = []
        # for data_z in data_n[:-1]:
        #    data.append(float(data_z))
        # i = 0
        # for data_z in data:
        #    ascii_ = data_z * 2
        #    decode_str += letters[int(ascii_)]
        #    i += 1
        # decode_str = decode_str.replace('-ou-', 'u')
        # decode_str = decode_str.split('@')
        # return decode_str
        return data

    @staticmethod
    def encode_code(data):
        # letters = string.ascii_letters + string.digits + string.punctuation
        # encode_str = ''
        # data = data.replace(' ', '@')
        # leng = data.__len__()
        # for data_st in range(leng):
        #    i = -1
        #    while data[data_st] != letters[i]:
        #        i += 1
        #        if data[data_st] == letters[i]:
        #            encode_str += str(i / 2) + '#'
        #    data_st += 1
        # return encode_str
        return str(data)
