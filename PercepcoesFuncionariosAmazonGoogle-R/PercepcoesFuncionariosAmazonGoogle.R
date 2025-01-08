
### Objetivo do Projeto
# Analisar e comparar as percepções de funcionários sobre as empresas Amazon e Google, identificando temas recorrentes nas avaliações positivas e negativas.  
# A análise visa destacar vantagens, desvantagens e associações textuais para melhor compreensão da cultura organizacional de ambas as empresas.  

# Imports
library(readr)
library(qdap)
library(tm)
library(RWeka)
library(wordcloud)
library(plotrix)
library(ggthemes)
library(ggplot2)

# Carga dos dados
df_amazon <- read_csv("dados/amazon.csv")
df_google <- read_csv("dados/google.csv")

# Visualiza os dados
View(df_amazon)
View(df_google)

# Tipos de dados
str(df_amazon)
str(df_google)

# Dimensões
dim(df_amazon)
dim(df_google)

# Prós e contras Amazon
amazon_pros <- df_amazon$pros
amazon_cons <- df_amazon$cons

# Prós e contras Google
google_pros <- df_google$pros
google_cons <- df_google$cons


### Pré-processamento

# Função para limpeza do texto
func_limpa_texto <- function(x){
  
  x <- na.omit(x)
  x <- replace_abbreviation(x)
  x <- replace_contraction(x)
  x <- replace_number(x)
  x <- replace_ordinal(x)
  x <- replace_symbol(x)
  x <- tolower(x)

  return(x)
}

# Aplicando a limpeza nos dados
amazon_pros <- func_limpa_texto(amazon_pros)
amazon_cons <- func_limpa_texto(amazon_cons)
google_pros <- func_limpa_texto(google_pros)
google_cons <- func_limpa_texto(google_cons)

# Converter o vetor contendo os dados de texto em um corpus volátil.
amazon_p_corp <- VCorpus(VectorSource(amazon_pros))
amazon_c_corp <- VCorpus(VectorSource(amazon_cons))
google_p_corp <- VCorpus(VectorSource(google_pros))
google_c_corp <- VCorpus(VectorSource(google_cons))

# Função de limpeza do Corpus
func_limpa_corpus <- function(x){
  
  x <- tm_map(x,removePunctuation)
  x <- tm_map(x,stripWhitespace)
  x <- tm_map(x,removeWords, c(stopwords("en"), "Amazon", "Google", "Company"))
  
  return(x)
}

# Aplica
amazon_pros_corp <- func_limpa_corpus(amazon_p_corp)
amazon_cons_corp <- func_limpa_corpus(amazon_c_corp)
google_pros_corp <- func_limpa_corpus(google_p_corp)
google_cons_corp <- func_limpa_corpus(google_c_corp)

# Tokenização
tokenizer <- function(x) 
NGramTokenizer(x, Weka_control(min = 2, max = 2))

# Feature extraction e análise de avaliações positivas
amazon_p_tdm    <- TermDocumentMatrix(amazon_pros_corp)
amazon_p_tdm_m  <- as.matrix(amazon_p_tdm)
amazon_p_freq   <- rowSums(amazon_p_tdm_m)
amazon_p_f.sort <- sort(amazon_p_freq, decreasing = TRUE)


# Prepara os dados para a wordcloud
amazon_p_tdm    <- TermDocumentMatrix(amazon_pros_corp, control = list(tokenize=tokenizer))
amazon_p_tdm_m  <- as.matrix(amazon_p_tdm)
amazon_p_freq   <- rowSums(amazon_p_tdm_m)
amazon_p_f.sort <- sort(amazon_p_freq,decreasing = TRUE)

# Cria o dataframe de comentários positivos
df_amazon_p <- data.frame(term = names(amazon_p_f.sort), num = amazon_p_f.sort)
View(df_amazon_p)

# Wordcloud
wordcloud(df_amazon_p$term, 
          df_amazon_p$num, 
          max.words = 100, 
          color = "tomato4")

# Feature extraction e análise de avaliações negativas
amazon_c_tdm    <- TermDocumentMatrix(amazon_cons_corp, control = list(tokenize = tokenizer))
amazon_c_tdm_m  <- as.matrix(amazon_c_tdm)
amazon_c_freq   <- rowSums(amazon_c_tdm_m)
amazon_c_f.sort <- sort(amazon_c_freq, decreasing = TRUE)

# Cria o dataframe de comentários negativos
df_amazon_c <- data.frame(term = names(amazon_c_f.sort), num = amazon_c_f.sort)
View(df_amazon_c)

# Wordcloud
wordcloud(df_amazon_c$term,
          df_amazon_c$num,
          max.words = 100,
          color = "palevioletred")

# Principais frases que apareceram nas wordcloud
amazon_p_tdm    <- TermDocumentMatrix(amazon_pros_corp, control = list(tokenize=tokenizer))
amazon_p_m      <- as.matrix(amazon_p_tdm)
amazon_p_freq   <- rowSums(amazon_p_m)
token_frequency <- sort(amazon_p_freq,decreasing = TRUE)
token_frequency[1:5]

# Associações
findAssocs(amazon_p_tdm, "fast paced", 0.2)

# Nuvem de palavras comparativa para as avaliações positivas e negativas do Google para comparação com a Amazon.
all_google_pros   <- paste(df_google$pros, collapse = "")
all_google_cons   <- paste(df_google$cons, collapse = "")
all_google        <- c(all_google_pros,all_google_cons)
all_google_clean  <- func_limpa_texto(all_google)
all_google_vs     <- VectorSource(all_google_clean) 
all_google_vc     <- VCorpus(all_google_vs)
all_google_clean2 <- func_limpa_corpus(all_google_vc)
all_google_tdm    <- TermDocumentMatrix(all_google_clean2)

# Colnames
colnames(all_google_tdm) <- c("Google Pros", "Google Cons")

# Converte para matriz
all_google_tdm_m <- as.matrix(all_google_tdm)

# Nuvem de comparação
comparison.cloud(all_google_tdm_m, colors = c("orange", "blue"), max.words = 50)


# Plot pirâmide alinhando comentários positivos
amazon_pro    <- paste(df_amazon$pros, collapse = "")
google_pro    <- paste(df_google$pros, collapse = "")
all_pro       <- c(amazon_pro, google_pro)
all_pro_clean <- func_limpa_texto(all_pro)
all_pro_vs    <- VectorSource(all_pro)
all_pro_vc    <- VCorpus(all_pro_vs)
all_pro_corp  <- func_limpa_corpus(all_pro_vc)

# Matriz termo-documento
tdm.bigram = TermDocumentMatrix(all_pro_corp,control = list(tokenize = tokenizer))

# Colnames
colnames(tdm.bigram) <- c("Amazon", "Google")

# Converte para matriz
tdm.bigram <- as.matrix(tdm.bigram)

# Palavras comuns
common_words <- subset(tdm.bigram, tdm.bigram[,1] > 0 & tdm.bigram[,2] > 0 )

# Diferença
difference <- abs(common_words[, 1] - common_words[,2])

# Vetor final
common_words <- cbind(common_words, difference)
common_words <- common_words[order(common_words[,3],decreasing = TRUE),]

# Dataframe
top25_df <- data.frame(x = common_words[1:25,1], 
                       y = common_words[1:25,2], 
                       labels = rownames(common_words[1:25,]))

# Plot
pyramid.plot(top25_df$x,
             top25_df$y,
             labels=top25_df$labels,
             gap=15,
             top.labels=c("Amazon Pros", "Vs", "Google Pros"),
             unit = NULL,
             main = "Palavras em Comum")


# Avaliações negativas com os mesmos recursos visuais.
amazon_cons    <- paste(df_amazon$cons, collapse = "")
google_cons    <- paste(df_google$cons, collapse = "")
all_cons       <- c(amazon_cons,google_cons)
all_cons_clean <- func_limpa_texto(all_cons)
all_cons_vs    <- VectorSource(all_cons)
all_cons_vc    <- VCorpus(all_cons_vs)
all_cons_corp  <- func_limpa_corpus(all_cons_vc)

# Matriz termo-documento
tdm.cons_bigram = TermDocumentMatrix(all_cons_corp,control=list(tokenize =tokenizer))

# Preparação dos dados 
colnames(tdm.cons_bigram) <- c("Amazon", "Google")
tdm.cons_bigram <- as.matrix(tdm.cons_bigram)
common_words <- subset(tdm.cons_bigram, tdm.cons_bigram[,1] > 0 & tdm.cons_bigram[,2] > 0 )
difference <- abs(common_words[, 1] - common_words[,2])
common_words <- cbind(common_words, difference)
common_words <- common_words[order(common_words[,3], decreasing = TRUE),]

# Dataframe
top25_df <- data.frame(x = common_words[1:25,1],
                       y = common_words[1:25,2],
                       labels = rownames(common_words[1:25,]))

# Plot
pyramid.plot(top25_df$x,
             top25_df$y,
             labels=top25_df$labels,
             gap=10,
             top.labels = c("Amazon Cons","Vs","Google Cons"),
             unit = NULL,
             main = "Palavras em Comum")


# Unigram
tdm.unigram <- TermDocumentMatrix(all_pro_corp)
colnames(tdm.unigram) <- c("Amazon","Google")
tdm.unigram <- as.matrix(tdm.unigram)

commonality.cloud(tdm.unigram, colors = c("tomato2", "yellow2"), max.words = 100)

# Bigram
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
tdm.bigram <- TermDocumentMatrix(all_pro_corp,control = list(tokenize=BigramTokenizer))
colnames(tdm.bigram) <- c("Amazon", "Google")
tdm.bigram <- as.matrix(tdm.bigram)

commonality.cloud(tdm.bigram, colors = c("tomato2", "yellow2"), max.words = 100)

# Trigram
TrigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))
tdm.trigram <- TermDocumentMatrix(all_pro_corp,control = list(tokenize=TrigramTokenizer))
colnames(tdm.trigram) <- c("Amazon","Google")
tdm.trigram <- as.matrix(tdm.trigram)

commonality.cloud(tdm.trigram, colors = c("tomato2", "yellow2"), max.words = 100)

# Palavras mais frequentes nos comentários dos funcionários
amazon_tdm <- TermDocumentMatrix(amazon_p_corp)
associations <- findAssocs(amazon_tdm,"fast",0.2)
associations_df <- list_vect2df(associations)[,2:3] 

ggplot(associations_df, aes(y = associations_df[,1])) +
  geom_point(aes(x = associations_df[,2]),
             data = associations_df, size = 3) + 
  theme_gdocs()

google_tdm <- TermDocumentMatrix(google_c_corp)
associations <- findAssocs(google_tdm,"fast",0.2)
associations_df <- list_vect2df(associations)[,2:3] 

ggplot(associations_df,aes(y=associations_df[,1])) +
  geom_point(aes(x = associations_df[,2]),
             data = associations_df, size = 3) + 
  theme_gdocs()


### Resultados e Conclusões  

# Comentários positivos sobre a Amazon frequentemente mencionam "bons benefícios", enquanto comentários negativos destacam o "equilíbrio trabalho-vida".  
# As vantagens do Google incluem "regalias", "boa comida" e "cultura divertida", enquanto as desvantagens apontam para "burocracia" e "política organizacional".  
# Ambos mencionam aspectos culturais e "pessoas inteligentes" como pontos positivos comuns.  
# A análise sugere que o Google apresenta uma percepção mais favorável quanto ao equilíbrio entre vida profissional e pessoal.  
# Este projeto fornece uma análise rica para insights organizacionais, que podem ser usados para iniciativas de retenção de talentos e aprimoramento cultural.


