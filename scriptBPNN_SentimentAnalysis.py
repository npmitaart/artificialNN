#import important libraries
import numpy
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split

#preprocessing function
def preprocessing(words, using_allowed_words=True):
    
    #lower function used to convert every words into lowercase
    words = words.lower()
    #split used to split the sentences into words, so each sentence will be easier to process
    words = words.split(' ')

    #init empty array
    allowed_words = []

    allowed_words = words

    clean_words = []
    
    #cleaning up every false sentences
    for word in words:
        if word in allowed_words:
            clean_words.append(word)

    return clean_words

#bag of words method used to extract all words from sentences
def bag_of_words(bag, words):
    bag_ofWords = []
    for b in bag:
        bag_ofWords.append(words.count(b))
    return bag_ofWords

#the activation function used is tanh
#most of time tanh is quickly converge than sigmoid and logistic function, and performs better accuracy
#why? the advantage is that the negative inputs will be mapped strongly negative and the zero inputs will be mapped near zero in the tanh graph
def tanh(value, derivative=False):
    if (derivative == True):
        return (1 - (x ** 2))
    return numpy.tanh(value)

#backpropagation for text classification
def generate_synapse(input_and_target, hidden_neurons = 3, alpha=0.1, epochs=10000, is_training=True):
    #init input list
    input_list = []
    #init output list
    output_list = []

    for x in input_and_target:
        input_list.append(x['input'])
        output_list.append(x['target'])
    
    input_list = numpy.array(input_list)
    output_list = numpy.array(output_list)

    if(is_training):
        print("Total Input  : %s" % len(input_list))
        print("Total Hidden Layers : %s" % hidden_neurons)
        
    #seed random numbers to make calculation (determenistic)
    numpy.random.seed(1)

    last_mean_error = 1

    #initial weight randomly
    syn0 = 2*numpy.random.random((len(input_and_target[0]['input']), hidden_neurons)) - 1
    syn1 = 2*numpy.random.random((hidden_neurons, len(input_and_target[0]['target']))) - 1

    #update previous weight
    syn0weight_update_prev = numpy.zeros_like(syn0)
    syn1weight_update_prev = numpy.zeros_like(syn1)

    #update weight for the next iteration (the newest)
    syn0weight_update_next = numpy.zeros_like(syn0)
    syn1weight_update_next = numpy.zeros_like(syn1)

    #iteration for update the best weight in backpropagation model
    for i in range(epochs+1):
        #init input layer
        layer_0 = input_list
        #fit tanh activation function in layer 1
        #forward propagation --> learning process
        layer_1 = tanh(numpy.dot(layer_0,syn0))
        #fit tanh activation function in layer 2
        #forward propagation --> learning process
        layer_2 = tanh(numpy.dot(layer_1,syn1))

        #calculate the error (how much the model miss?)
        error_layer_2 = output_list - layer_2
        #multiply the error
        delta_layer_2 = error_layer_2 * tanh(layer_2)

        error_layer_1 = delta_layer_2.dot(syn1.T)
        delta_layer_1 = error_layer_1 * tanh(layer_1)

        if (i% 1000) == 0:
            err_iteration = numpy.mean(numpy.abs(error_layer_2))
            if err_iteration < last_mean_error:
                print ("delta after "+str(i)+" iterations:" + str("{:10f}".format(err_iteration))+ " or "+str("{:10f}".format(1-err_iteration)))
                last_mean_error = err_iteration
            else:
                print ("break:", err_iteration, ">", last_mean_error )
                break
        
        #update weight
        syn1weight_update = (layer_1.T.dot(delta_layer_2))
        syn0weight_update = (layer_0.T.dot(delta_layer_1))
        
        if(i>0):
            syn0weight_update_next += numpy.abs(((syn0weight_update > 0)+0) - ((syn0weight_update_prev > 0) + 0))
            syn1weight_update_next += numpy.abs(((syn1weight_update > 0)+0) - ((syn1weight_update_prev > 0) + 0)) 

        syn1 += alpha * syn1weight_update
        syn0 += alpha * syn0weight_update
        
        syn0weight_update_prev = syn0weight_update
        syn1weight_update_prev = syn1weight_update

    #converting data in synapse to list type
    synapse = {
        'synapse_0' : syn0.tolist(),
        'synapse_1' : syn1.tolist()
    }

    return synapse

#training process
def generate_training(training_datas, hidden_neurons = 3, alpha=0.1, epochs=10000, use_allowed_words=True, is_debugging=True):
    classes = []
    words = []

    #print total training data
    print("Total training data: %s"%len(training_datas))
    
    #do the preprocessing step to the input data
    for data in training_datas:
        data['values'] = preprocessing(data['value'], use_allowed_words)
        if(is_debugging):
            print(data['values'])
            print("\n")
            
    #show the output classes
    for data in training_datas:
        if data['class'] not in classes:
            classes.append(data['class'])
        for w in data['values']:
            if w not in words:
                words.append(w)
    
    if(is_debugging):
        #print the total classes
        print("Total classes: ", len(classes))
        print(classes)
        print("\n")
        #print total words processed
        print("Total words: ", len(words))
        print(words)
        print("\n")

    input_and_target = []

    for data in training_datas:
        documentBow = bag_of_words(words, data['value'])
        documentClass = []
        for c in classes:
            documentClass.append(1) if c==data['class'] else documentClass.append(0)
        
        print("\n")
        print("Bag of Words")
        print(documentBow)
        print("Class")
        print(documentClass)
        print("\n")

        input_and_target.append({
            "target" : documentClass,
            "input" : documentBow
        })
    
    return {
        'words': words,
        'classes': classes,
        'use_allowed_words': use_allowed_words,
        'input_and_target' : input_and_target,
        'synapse' : generate_synapse(input_and_target,hidden_neurons,alpha,epochs,is_debugging)
    }

#classification process
def classify(training, sentence, is_debugging=True):
    wordList = preprocessing(sentence, training['use_allowed_words'])
    classList = training['classes']
    trainWordList = training['words']

    bag_ofWords = []
    for x in trainWordList:
        bag_ofWords.append(wordList.count(x))

    #set bag of words to array
    bag_ofWords = numpy.array(bag_ofWords)

    layer_0 = bag_ofWords 
    layer_1 = tanh(numpy.dot(layer_0,training['synapse']['synapse_0']))
    layer_2 = tanh(numpy.dot(layer_1,training['synapse']['synapse_1']))

    print("\n\n")
    print("Classify")
    print(wordList)
    print(layer_2)

    response = []

    i = 0
    for x in classList:
        response.append({
            'class': x,
            'value': "{:10f}".format(layer_2[i])
        })
        i = i+1

    return response

#data to classify
training_datas = [
    {
        'class':'positive',
        'value':'This was a great movie with a good cast, all of them hitting on all'
    },
    {
        'class':'negative',
        'value':'Even if you are a huge Sandler fan, please do not bother with this'
    },
    {
        'class':'positive',
        'value':'A movie of outstanding brilliance and a poignant and unusual love story'
    },
    {
        'class':'negative',
        'value':' I had the misfortune to watch this rubbish on Sky Cinema Max in a cold'
    },
    {
        'class':'negative',
        'value':'I am at a distinct disadvantage here. I have not seen the first two movies'
    },
    {
        'class':'negative',
        'value':'This program is a lot of fun and the title song is so catchy I can not get it'
    }
]

#training process
training = generate_training(training_datas, 3, 0.1, 10000, False, True)

classify_results = []

for x in training_datas:
    classify_results.append(classify(training,x['value']))

print("\n")
print("Testing result:")
for x in classify_results:
    print(x)
    
print("\n")
print("Classify Final result:")
classify(training,"This was a great movie with a good cast, all of them hitting on all")
classify(training,"Even if you are a huge Sandler fan, please do not bother with this")
classify(training,"A movie of outstanding brilliance and a poignant and unusual love story")
classify(training,"I had the misfortune to watch this rubbish on Sky Cinema Max in a cold")
classify(training,"I am at a distinct disadvantage here. I have not seen the first two movies")
classify(training,"This program is a lot of fun and the title song is so catchy I can not get it")
