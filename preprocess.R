data <- read.csv('dataset_diabetes/diabetic_data.csv', stringsAsFactors=F)

# ??? Change from 3-class classification to binary classification ???
target.column <- 'readmitted'
non.predictive.columns <- c('encounter_id', 'patient_nbr')

# These columns have > 50% missing data
dropped.columns <- c('weight', 'payer_code', 'medical_specialty')

data <- data[!(names(data) %in% non.predictive.columns)]
data <- data[!(names(data) %in% dropped.columns)]

# Replace factors with integer levels
feature.names <- names(data)
for (f in feature.names) {
	if (class(data[[f]]) == "character") {
		levels <- unique(data[[f]])
		data[[f]] <- as.integer(factor(data[[f]], levels=levels))
	}
}

# Shuffle
num.examples <- nrow(data)
set.seed(123)
data <- data[sample(num.examples),]

# Set train:test split of 80:20
split <- as.integer(0.8 * num.examples)
train <- data[1:split,]
test <- data[split:num.examples,]

# Output {train,test}_{x,y}.csv
write.csv(train[names(train) != target.column], 'train_x.csv', row.names=F)
write.csv(train[names(train) == target.column], 'train_y.csv', row.names=F)
write.csv(test[names(test) != target.column], 'test_x.csv', row.names=F)
write.csv(test[names(test) == target.column], 'test_y.csv', row.names=F)
