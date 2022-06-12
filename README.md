# character_relationship_analysis_pipeline
End-to-end pipeline for extracting and analysing interactions between pairs of characters in books.

### Step 1 - Character Identification 

Character names are identified using SpaCy's pre-trained Named Entity Recognition model. I filter for the top 20 most mentioned character names. Characters included twice, in the forms ‘forename-surname’ and ‘forename’ are resolved to a single character entity, while character names in the possessive form (e.g. "Harry's") are
reduced to names only. 

### Step 2 - Coreference Resolution

I train a new coreference resolution model, Char-Coref, on an adapted version of the Litbank Coreference Resolution dataset. The model is based on Long-Doc Coref (Toshniwal et al. (2020)), but is adapted to be able to perform inference on documents 10x larger than the original model allowed. The dataset on which it is trained is identical to the orignial Litbank dataset except for the fact that all non-person entities were removed so that the model only picks up references to characters.

This model identifies all references (names, pronouns, etc) to characters in the novel. Each entity (character) gets a cluster of mentions that spans the length of their involvement in the text. These clusters are then matched with one character name from the results of Step 1.

### Step 3 - Character interaction extraction

I define an 'interaction' between two characters as any sentence containing references to both characters. Using each character's respective mention cluster, I find all sentences containing interactions. 

### Step 4 - Interaction Sentiment Analysis and Results Presentation

Each pair of characters now has a sequence of interactions, indexed by their positions in the text. Using rule-based sentiment analyser VADER, I assign a sentiment score between -1 and 1 to each interaction, and plot the cumulative sentiment scores for each pair of characters, producing a time series plot for any given model, such as the one shown below for Harry Potter and the Philosopher's Stone. 





![0](https://user-images.githubusercontent.com/76516724/173229267-1d3e3786-9fcb-427f-838d-8ee050651d54.jpg)
