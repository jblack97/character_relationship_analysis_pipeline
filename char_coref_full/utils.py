#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import itertools
import json
import sys
import pickle
import re

from inference.inference import Inference

# In[2]:


"""
inverse mapping of tokenized text to list text
Input: doc_tokens (['B', '##O', '##O', '##K', ...])
Output: ['BOOK', ...]
"""
def inv_map(tokenized_text):
  text = []
  for i in tokenized_text:
    if len(i) < 3:
      text.append(i)
    elif i[0:2] != '##':
      text.append(i)
    else:
      text[-1] = text[-1] + i[2:]
  return text


# In[3]:


"""
Input: Document in list form, including punctuation as elements of list
Output: List of indexes of all sentence separating punctuation ('.' '?' '!')
e.g [1,30,33,40,...]
Can easily be adapted to include ','
"""

def punctuation_indices(document):
  indices = [i for i, x in enumerate(document) if x == '.' or x == '?' or x == '!' or x == ';']
  prenames = ['Dr', 'Mr', 'Mrs', 'Miss', 'Ms']
  false_indices = []
  for i in range(len(document)):
    if document[i] == '.':
      if document[i-1] in prenames:
        false_indices.append(i)
  removals = set(false_indices)
  indices = [x for x in indices if x not in removals]
  return indices


# In[4]:


"""
Input: list of sentence separating punctuation indices 
and clusters (entity id, and all its mentions' indices), e.g. [[(3,4), (130,133), 1],[(5,5), (399,399), 2],...]
Output: Augmented list of punctuation indices (includes the mention indices in between the punctuation indices,
and has the corresponding entity id in the 2nd column)
e.g. [[2,0],[3,1],[49,0],[87,0],[130,1],...]
"""
def add_entity_indices(punc_indices, clusters):
  augmented_list = punc_indices.copy()
  for i in range(len(augmented_list)):
    augmented_list[i] = [augmented_list[i],0]

  index_additives = np.zeros(len(punc_indices))

  for cluster in clusters:
    starting_index = 0
    for i in range(len(cluster)-1):
      for j in range(len(punc_indices[starting_index:])-1):
        if cluster[i][0] > punc_indices[starting_index + j] and cluster[i][0] < punc_indices[starting_index + j+1]:
          index_additives[starting_index + j:] += 1
          augmented_list.insert(starting_index + j + int(index_additives.item(starting_index + j)), [cluster[i][0],[cluster[-1], cluster[i]]])
          starting_index += j 
          break
  return np.asarray(augmented_list)


# In[5]:


"""
Input: Augmented list of punctuation indices with character mentions and characters 
(represented by number, e.g. Achilles = 1, agamemnon = 2)
Output: Shared sentence indices and corresponding character numbers included in that sentence, 
e.g [[(2,9), [1,2]], [(30,45), [1,1,2]], [(50,67), [3,3]],...]
"""

def shared_sentences(indices):
  characters = []
  zeros = np.where(indices == 0)[0]
  for i in range(np.shape(indices[:,1])[0]-1):
    if indices[:,1][i] == 0 and indices[:,1][i+1] != 0 and indices[:,1][i+2] != 0:
      characters.append([(indices.item(i,0)+1 , indices.item(zeros[np.where(zeros == i)[0][0] + 1],0)+1) , indices[i+1:zeros[np.where(zeros == i)[0][0] + 1],1].tolist()])
  return characters


# In[6]:


"""
removes mentions contained in other mentions
"""
def clean_1(share):
  for i in range(len(share)):
    sort = share[i][1]
    sort.sort(key=lambda j: (j[1][0], -j[1][1]),reverse=False)
    output = [sort[0]]
    #sort = np.asarray(sort)[:,1]
    for num, r in sort[1:]:
      prevL, prevR = output[-1][1]
      if prevL <= r[0] and prevR >= r[1]:
        continue
      output.append([num,(r[0],r[1])])
    share[i][1] = output
  return share


"""
Input: List of 'shared' sentence beginning and end indices, with corresponding
numbers representing the characters involved in the sentence
Output: Removes all entries where the same character appears more than once
with no other character in the sentence, e.g. [[(2,9), [1,2]], [(30,45), [1,1,2]],...]
"""

def remove_same(sentences):
  remove = []
  count = 0
  for i in sentences:
    if len(list(set(np.asarray(i[1])[:,0]))) == 1:
      remove.append(count)
    count += 1
  remove.reverse()
  for i in remove:
    sentences.pop(i)
  return sentences


# In[7]:


"""
convert list text to normal document text
"""

def list_to_doc(text):
  string = ' '.join(word for word in text)
  string = string.replace(' .', '.')
  string = string.replace(' ,', ',')
  string = string.replace(' !', '!')
  string = string.replace(' ?', '?')
  string = string.replace(' ;', ';')
  string = string.replace(' :', ':')
  string = string.replace('[ ', '[')
  string = string.replace(' ]', ']')
  string = string.replace(' )', ')')
  string = string.replace('( ', '(')
  string = string.replace(" ’ ", "’")
  string = string.replace(" ' ", "'")
  string = string.replace(" - ", "-")
  string = string.replace("s’", "s’ ")
  string = string.replace("\u201c ", "\u201c")
  string = string.replace(" \u201d ", "\u201d")
  return string

# In[9]:


"""
Input: dictionary of character names with corresponding number representing character, list of clusters, original list (not tokenized)
Output: List of clusters, streamlined so there are only mention indices and 1 number representing character
can account for when multiple clusters refer to the same character
uses dictionaries instead of lists, more efficient
Characters: Dictionary (key = name, value = number)
Output: Dictionary [[(indices), (indices),..., id-number], ..., ...]
"""
def augment_clusters(clusters, characters):
  augmented_clusters = {}
  for cluster in clusters:
    check_in = False
    check_first = False
    for i in cluster:
      if i[1] in characters:
        check_in = True
        number = characters[i[1]]
        #character = i[1]
        if number not in augmented_clusters:
          check_first = True
        break
    if check_in:
      if check_first:
        augmented_clusters[number] = [i[0] for i in cluster]
      else:
        augmented_clusters[number].extend([i[0] for i in cluster])

  for i in augmented_clusters:
    augmented_clusters[i] = list(set(augmented_clusters[i]))
    augmented_clusters[i].sort(key=lambda j: (j[0], -j[1]),reverse=False)
    output = [augmented_clusters[i][0]]
    for l, r in augmented_clusters[i][1:]:
      prevL, prevR = output[-1]
      if prevL <= l and prevR >= r:
        continue
      output.append((l,r))
    augmented_clusters[i] = output

  list_augmented_clusters = list(augmented_clusters.items())

  def tuple_to_list(tupl):
    new_list = [i for i in tupl]
    return new_list

  """def sort_tuples(key_value):
          key_value[1].sort(key=lambda i:i[0],reverse=False)
          return key_value"""

  def start_to_end(some_list):
    new_list = some_list[1]
    new_list.append(some_list[0])
    return new_list

  augmented_list = [start_to_end(tuple_to_list(i)) for i in list_augmented_clusters]
  
  return augmented_list


"""
same as above, but assigns clusters to most occuring character
"""
def augment_clusters_new(clusters, characters):
  augmented_clusters = {}
  for cluster in clusters:
    tallies = dict.fromkeys(characters.keys(),0)
    for i in cluster:
      if i[1] in characters:
        tallies[i[1]] += 1
    char = max(tallies, key = tallies.get)
    number = characters[char]
    if number in augmented_clusters:
      augmented_clusters[number].extend([i[0] for i in cluster])
    else:
      augmented_clusters[number] = [i[0] for i in cluster]

  for i in augmented_clusters:
    augmented_clusters[i] = list(set(augmented_clusters[i]))
    augmented_clusters[i].sort(key=lambda j: (j[0], -j[1]),reverse=False)
    output = [augmented_clusters[i][0]]
    for l, r in augmented_clusters[i][1:]:
      prevL, prevR = output[-1]
      if prevL <= l and prevR >= r:
        continue
      output.append((l,r))
    augmented_clusters[i] = output

  list_augmented_clusters = list(augmented_clusters.items())

  def tuple_to_list(tupl):
    new_list = [i for i in tupl]
    return new_list


  def start_to_end(some_list):
    new_list = some_list[1]
    new_list.append(some_list[0])
    return new_list

  augmented_list = [start_to_end(tuple_to_list(i)) for i in list_augmented_clusters]
  
  return augmented_list


# In[10]:


"""
Input: list of characters (their numbers and name), [[id, character], ...]
output: character pair encoding dictionary and corresponding character pairs for reference
and a dictionary of character pair numbers with empty list as their value
dictionary_1 {5: (Achilles, Agamemnon), ...}
dictionary_2 {5 : [], ...}
"""
def character_pair_encoder(characters):
  n = len(characters)
  numbers = list(range(n+1))
  numbers.remove(0)

  dictionary_1 = {}
  dictionary_2 = {}

  for i in numbers:
    for j in numbers[i:]:
      dictionary_1[i**2 + j **2] = (characters[i-1][1], characters[j-1][1])
      dictionary_2[i**2 + j **2] = []

  return dictionary_1, dictionary_2


def character_pair_encoder_new(characters):
  n = len(characters)
  numbers = list(range(n+1))
  numbers.remove(0)

  dictionary_1 = {}
  dictionary_2 = {}

  for i in numbers:
    for j in numbers[i:]:
      dictionary_1[str(sorted([i,j]))] = (characters[i-1][1], characters[j-1][1])
      dictionary_2[str(sorted([i,j]))] = []

  return dictionary_1, dictionary_2


# In[11]:


"""
Goes through the shared sentences with their shared character ids
Assigns the sentence indices to a dictionary of all character pairs based on the encoding above
Input: shared - , pair_dict - 
Output: 
"""

def assign_to_dict(shared, pair_dict):
  for i in shared:
    if len(i[1]) == 2:
      pair_dict[i[1][0][0]**2 + i[1][1][0]**2].extend([i])
    else:
      for a in list(itertools.combinations(set(np.asarray(i[1])[:,0]), 2)):
        pair_dict[a[0]**2 + a[1]**2].extend([i])
  return pair_dict


def assign_to_dict_new(shared, pair_dict):
  for i in shared:
    if len(i[1]) == 2:
      pair_dict[str(sorted([i[1][0][0], i[1][1][0]]))].extend([i])
    else:
      for a in list(itertools.combinations(set(np.asarray(i[1])[:,0]), 2)):
        pair_dict[str(sorted([a[0], a[1]]))].extend([i])
  return pair_dict


# In[ ]:


"""
function to give relative list index of character in sentence
Input: sentence indices, character indices in entire document
Output: character indices relative to sentence
"""

def rel_indices(sentence_inds, char_inds):
  return (char_inds[0] - sentence_inds[0], char_inds[1]- sentence_inds[0])


"""
CHANGE
Converts a pair of list indices to a pair of corresponding string indices
"""
def list_to_str_index(doc, indices):
  return (len(list_to_doc(doc[:indices[0]+1]))-len(doc[indices[0]]), len(list_to_doc(doc[:indices[1]+1])))

"""
CHANGE
Function that given the sentence list indices, 
and the character mention indices relative to the sentence
Returns the sentence as a string, and the character mention
indices as string indices
"""
def convert_dict(dic, document, additor):
  for i in dic:
    for n in  range(len(dic[i])):
      list_text = document[dic[i][n][0][0]: dic[i][n][0][1]]
      text = list_to_doc(list_text)
      
      dic[i][n] = [text, [[a[0], list_to_str_index(list_text, rel_indices(dic[i][n][0], a[1])), 
                           text[list_to_str_index(list_text, rel_indices(dic[i][n][0], a[1]))[0]:list_to_str_index(list_text, rel_indices(dic[i][n][0], a[1]))[1]]] for a in dic[i][n][1]], dic[i][n][0][0] + additor]
  return dic


"""
converts the dictionary of list_document indices above to
dictionary of shared sentences in string format
"""

def dict_ind_to_sentence(dictionary, document):
  for i in dictionary:
    dictionary[i] = [list_to_doc(document[a[0]:a[1]]) for a in dictionary[i]]
  return dictionary


"""
removes all pairs from dictionary that have no shared sentences
"""
def remove_empty(dic):
  return {k:v for k,v in dic.items() if v}


"""
Input: sentence, list of indices of portion to replace with what to replace with
       character id dictionary {key = id : value = character name}
output: sentence with mentions replaced by corresponding entity
"""
def replace(cluster, char_id_dic):
  sort = cluster[1]
  sentence = cluster[0]
  sort.sort(key=lambda i:i[1][0],reverse=False)

  sentences = [sentence[:sort[0][1][0]]]
  for i in range(len(sort)-1):
    sentences.append(char_id_dic[sort[i][0]])
    sentences.append(sentence[sort[i][1][1]:sort[i+1][1][0]])
  sentences.append(char_id_dic[sort[-1][0]])
  sentences.append(sentence[sort[-1][1][1]:]) 

  return ''.join(sentences)


"""
Input: dictionary of pairs, with shared sentences and corresponding mentions/entities/indices
Output: dictionary of pairs, with shared sentences where mention is replaced by entity 
"""
def alter_dict(dic, char_id_dic):
  id_dic = dict((v,k) for k,v in char_id_dic.items())
  for i in dic:
    total = []
    for n in dic[i]:
      total.append(replace(n,id_dic) + '\n')
    dic[i] = ''.join(total)
  return dic


"""
same as above, but keeps sentence start index, and doesn't return block of text, but list 
"""
def new_alter_dict(dic, char_id_dic):
  id_dic = dict((v,k) for k,v in char_id_dic.items())
  for i in dic:
    total = []
    for n in dic[i]:
      total.append([n[-1],replace(n,id_dic)])
    dic[i] = total
  return dic


"""
given the shared sentences, returns dictionary with just the unaltered shared sentences
"""
def alter_dict_noreplace(dic):
  for i in dic:
    total = []
    for n in dic[i]:
      total.append([n[-1], n[0]])
    dic[i] = total
  return dic


def calculate_additor(start_ind, doc_tokens):
	return len(inv_map(doc_tokens)[:start_ind])

"""
put all together to output pairs with shared sentences with mentions replaced
by entity
"""
def bigfunc_with_replace(clusters, characters_dict, characters_list, list_text, additor):
  augmented_clusters = augment_clusters_new(clusters, characters_dict)
  punc_indices = punctuation_indices(list_text)
  augmented_indices = add_entity_indices(punc_indices, augmented_clusters)
  shared = shared_sentences(augmented_indices)
  shared = clean_1(shared)
  cleaned_shared = remove_same(shared)

  encoding_dict, shared_sentence_dict = character_pair_encoder_new(characters_list)
  pair_dict = assign_to_dict_new(cleaned_shared, shared_sentence_dict)
  #pair_dict = remove_empty(pair_dict)
  pair_dict = convert_dict(pair_dict, list_text, additor)
  pair_dict = new_alter_dict(pair_dict, characters_dict)

  encoding_json = json.dumps(encoding_dict, indent = 4)
  #pair_json = json.dumps(pair_dict, indent = 4)

  return encoding_json, pair_dict.copy()


"""
same as above, but doesn't replace mentions with entity, and keeps
information of where the mentions are within the shared sentences
and also what the mentions are
"""
def bigfunc_no_replace(clusters, characters_dict, characters_list, list_text, additor):
  augmented_clusters = augment_clusters_new(clusters, characters_dict)
  punc_indices = punctuation_indices(list_text)
  augmented_indices = add_entity_indices(punc_indices, augmented_clusters)
  shared = shared_sentences(augmented_indices)
  shared = clean_1(shared)
  cleaned_shared = remove_same(shared)

  encoding_dict, shared_sentence_dict = character_pair_encoder_new(characters_list)
  pair_dict = assign_to_dict_new(cleaned_shared, shared_sentence_dict)
  #pair_dict = remove_empty(pair_dict)
  pair_dict = convert_dict(pair_dict, list_text, additor)
  pair_dict = alter_dict_noreplace(pair_dict)

  encoding_json = json.dumps(encoding_dict, indent = 4)
  #pair_json = json.dumps(pair_dict, indent = 4)

  return encoding_json, pair_dict.copy()


"""
Function that given list of dictionaries with same keys, amalgamates values chronologically,
and doesn't including extra overlapping sentences
"""
def amalgamate(dics):
  for i in range(len(dics)-1):
    for j in dics[i]:
      if dics[i][j] == []:
        continue
      if dics[i+1][j] == []:
        continue
      value = dics[i][j][-1][0]
      for k in range(len(dics[i+1][j])):
        if dics[i+1][j][k][0] > value:
          dics[i+1][j] = dics[i+1][j][k+1:]
          break
        if k == len(dics[i+1][j]):
          dics[i+1][j] = []
          break

  for j in dics[0]:
    for i in range(len(dics)-1):
      dics[0][j].extend(dics[i+1][j])
  
  pair_json = json.dumps(remove_empty(dics[0]), indent = 4)

  return pair_json  



def get_char_ids(char_list):

  #Dictionary of character IDs
  char_dict = {}

  for id, char in enumerate(char_list):
    char_dict[char] = id + 1
  
  return char_dict

def char_id_list(char_list):
    char_list = [0] + char_list
    char_list = [[i,a] for i,a in enumerate(char_list)]
    char_list.pop(0)
    return char_list




def do_coreference(book, doc, directory):
  #!pip install -U -q PyDrive
  from pydrive.auth import GoogleAuth
  from pydrive.drive import GoogleDrive
  from google.colab import auth
  from oauth2client.client import GoogleCredentials
  # Authenticate and create the PyDrive client.
  # This only needs to be done once per notebook.
  auth.authenticate_user()
  gauth = GoogleAuth()
  gauth.credentials = GoogleCredentials.get_application_default()
  drive = GoogleDrive(gauth)
  file_id = '1tNqhCbAE4DK7U2b9UUvElI6TRkTUwP63' # URL id. 
  downloaded = drive.CreateFile({'id': file_id})
  downloaded.GetContentFile('Copy of model.pth')

  doc = re.sub('\n', ' ', doc)
  doc = re.sub('\'', '’', doc)
  doc = re.sub(' \"', ' \u201c', doc)
  doc = re.sub('\" ', '\u201d ', doc)

  inference_model = Inference('/content/Copy of model.pth')

  if book == 1:
    portion_1 = doc[:200000]
    output = inference_model.perform_coreference(portion_1)
    output_1 = output["clusters"]

    portion_2 = doc[199900:]
    output = inference_model.perform_coreference(portion_2)
    output_2 = output["clusters"]

    clusters = [output_1,output_2]
    
  elif book == 3:
    portion_1 = doc[:200000]
    output = inference_model.perform_coreference(portion_1)
    output_1 = output["clusters"]

    portion_2 = doc[199900:400000]
    output = inference_model.perform_coreference(portion_2)
    output_2 = output["clusters"]

    portion_3 = doc[399900:600000]
    output = inference_model.perform_coreference(portion_3)
    output_3 = output["clusters"]

    portion_4 = doc[699900:]
    output = inference_model.perform_coreference(portion_4)
    output_4 = output["clusters"]

    clusters = [output_1,output_2,output_3,output_4]
  
  else:
    portion = doc
    output = inference_model.perform_coreference(portion)
    output = output["clusters"]

    clusters = [output]

  return clusters




def get_shared_sentences(book, doc, chars, use_own = False, cluster_list = []):
  if book == 1:
    files = ['Book_1_new[_200000]', 'Book_1_new[199900_]']
    ranges = [(0,200000), (199900,-1)]

  if book == 3:
    files = ['dracula[_200000]', 'dracula[199900_400000]','dracula[399900_600000]', 'dracula[599900_]']
    ranges = [(0,200000), (199900,400000), (399900,600000), (599900,-1)]
    
  if book == 4:
    files = ['chocolate_factory']
    ranges = [(0,-1)]
    
  if book == 2:
    files = ['peter_pan']
    ranges = [(0,-1)]
  
  if book == 5:
    files = ['winnie_the_pooh']
    ranges = [(0,-1)]

  else:
    ranges = [(0,-1)]
  
  doc = re.sub('\n', ' ', doc)
  doc = re.sub('\'', '’', doc)
  doc = re.sub(' \"', ' \u201c', doc)
  doc = re.sub('\" ', '\u201d ', doc)

  char_dict = get_char_ids(chars)
  char_list = char_id_list(chars)

  dics = []
  for i in range(len(ranges)):
    portion = doc[ranges[i][0]:ranges[i][1]]
    pre_portion = doc[:ranges[i][0]]
    
    tokens = get_tokenized_doc(portion, tokenizer)
    doc_tokens = flatten(tokens['sentences'])
    pre_tokens = get_tokenized_doc(pre_portion, tokenizer)
    pre_doc_tokens = flatten(pre_tokens['sentences'])
    
    document = inv_map(doc_tokens)

    additive = calculate_additor(ranges[i][0], pre_doc_tokens)

    
    if use_own:
      clusters = [j for j in cluster_list[i]]
    else:
      with open(f"character_relationship_analysis/data/newest clusters/{files[i]}", "rb") as fp:
        clusters = [j for j in pickle.load(fp)]
    
    encoding_dict, sentences = bigfunc_with_replace(clusters, char_dict, char_list, document, additive)
    dics.append(sentences)

  pair_dict = amalgamate(dics)

  with open("content/encoding.json", "w") as fp:
    json.dump(encoding_dict, fp)

  with open("content/pair_replace.json", "w") as fp:
    json.dump(pair_dict, fp)

  #return encoding_dict, amalgamate(dics)