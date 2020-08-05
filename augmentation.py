from textattack.augmentation import EmbeddingAugmenter

augmenter = EmbeddingAugmenter(pct_words_to_swap=0.2,transformations_per_example=2)

with open('dataset/test.txt','r') as r,open('augmentation_test.txt','w') as w:
    for line in r:
        list_labels_id = line.split(';')
        aug = augmenter.augment(list_labels_id[0])
        w.write(list_labels_id[0]+';'+list_labels_id[1])
        for aug_text in aug:
            w.write(aug_text+';'+list_labels_id[1])
        
