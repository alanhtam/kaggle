library(tm)
library(caret)
library(randomForest)

# load training and test data
eBayTrain = read.csv("eBayiPadTrain.csv", stringsAsFactors=FALSE)
eBayTest = read.csv("eBayiPadTest.csv", stringsAsFactors=FALSE)

# process data for cross validation
trainIndex = createDataPartition(eBayTrain$sold, p=0.50, list=FALSE)
train1 = eBayTrain[ trainIndex,]
train2 = eBayTrain[-trainIndex,]

# create corpus
CorpusDescription = Corpus(VectorSource(c(train1$description, train2$description, eBayTest$description)))

# preprocess corpus
CorpusDescription = tm_map(CorpusDescription, content_transformer(tolower), lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, PlainTextDocument, lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, removePunctuation, lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, removeWords, stopwords("english"), lazy=TRUE)
CorpusDescription = tm_map(CorpusDescription, stemDocument, lazy=TRUE)

#convert to dataframe and remove sparse terms
dtm = DocumentTermMatrix(CorpusDescription)
sparse = removeSparseTerms(dtm, 0.995)
DescriptionWords = as.data.frame(as.matrix(sparse))

# check colname R compatibility
colnames(DescriptionWords) = make.names(colnames(DescriptionWords))

# split back into training and test sets
DescriptionWordsTrain1 = head(DescriptionWords, nrow(train1))
DescriptionWordsTrain2 = DescriptionWords[(nrow(train1)+1):(nrow(train1)+nrow(train2)),]
DescriptionWordsTest = tail(DescriptionWords, nrow(eBayTest))

# add relevant independent variables
DescriptionWordsTrain1$sold = train1$sold
DescriptionWordsTrain1$startprice = train1$startprice
DescriptionWordsTrain1$condition = train1$condition
DescriptionWordsTrain1$biddable = train1$biddable
DescriptionWordsTrain1$cellular = train1$cellular
DescriptionWordsTrain1$productline = train1$productline
DescriptionWordsTrain1$wordcount = sapply(gregexpr("\\W+", train1$description), length) + 1

DescriptionWordsTrain2$sold = train2$sold
DescriptionWordsTrain2$startprice = train2$startprice
DescriptionWordsTrain2$condition = train2$condition
DescriptionWordsTrain2$biddable = train2$biddable
DescriptionWordsTrain2$cellular = train2$cellular
DescriptionWordsTrain2$productline = train2$productline
DescriptionWordsTrain2$wordcount = sapply(gregexpr("\\W+", train2$description), length) + 1

DescriptionWordsTest$startprice = eBayTest$startprice
DescriptionWordsTest$condition = eBayTest$condition
DescriptionWordsTest$biddable = eBayTest$biddable
DescriptionWordsTest$cellular = eBayTest$cellular
DescriptionWordsTest$productline = eBayTest$productline
DescriptionWordsTest$wordcount = sapply(gregexpr("\\W+", eBayTest$description), length) + 1

DescriptionWordsLog1 = glm(sold ~ scratch + power + owner + ipad +
                            startprice + condition + biddable + cellular + still +
                            productline + wordcount, 
                          DescriptionWordsTrain1, 
                          family=binomial)

# create prediction object on test data
PredTest = predict(DescriptionWordsLog1, newdata=DescriptionWordsTrain2, type="response")
table(DescriptionWordsTrain2$sold, PredTest > 0.5)

# format data frames for submission
MySubmission = data.frame(UniqueID = eBayTest$UniqueID, Probability1 = PredTest)
write.csv(MySubmission, "SubmissionDescriptionLog.csv", row.names=FALSE)
