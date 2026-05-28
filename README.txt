Le cartelle contengono le 3 componenti del dataset:

1) "bioactivities" contiene l'elenco di bioattività tra molecole e proteine target. Source Papyrus: https://pmc.ncbi.nlm.nih.gov/articles/PMC9824924/)

Identificatori:
	bioattività --> "Activity_ID"
	molecole    --> "InChIKey"
	proteine    --> "target_id"
	pChembl Mean/Median --> valore di bioattività. Di solito si considera pChembl > 5.5 come interazione attiva (Active) e pChembl <=5.5 come interazione inattiva (Inactive). E' una classificazione euristica, quindi nulla di definito. Altri usano una suddivisione di 
	pChembl < 5.0 inactive
	5.0 <= pChembl < 6.0 borderline
	pChembl >= 6.0 active
	
2) "mass_spectra_embeddings" contiene gli embedding delle molecole ottenuti con il tool python Spec2Vec (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008724)

"ALL_GNPS_negative_ionization_DATASET.csv" --> dataset di embedding di molecole ottenute con il tool Spec2Vec a partire dagli spettri di massa.

"Pairwise_similarity_scores.tsv" --> metriche di qualità degli embedding ottenuti con Spec2Vec usando cosine similarity.

"spec2vec_vs_baseline.png" --> plot delle metrichè di qualità degli embedding per inter ed intra similarità usando cosine similarity. 

3) "protein_embeddings" contiene gli embedding delle proteine ottenuti con UniRep. Source Papyrus: https://pmc.ncbi.nlm.nih.gov/articles/PMC9824924/

Ci sono 3 file, ognuna contenente una rappresentazione delle proteine usando embedding di dimensione diversa.

File:"protein_embeddings_UniRep64.csv"
- embedding costituito da 3 componenti da 64 features/colonne, per un totale di 192 features

File:"protein_embeddings_UniRep256.csv"
- embedding costituito da 3 componenti da 256 features/colonne, per un totale di 768 features

File:"protein_embeddings_UniRep1900.csv"
- embedding costituito da 3 componenti da 1900 features/colonne, per un totale di 5700 features

