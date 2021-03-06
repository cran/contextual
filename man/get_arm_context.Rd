% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/functions_utility.R
\name{get_arm_context}
\alias{get_arm_context}
\title{Return context vector of an arm}
\usage{
get_arm_context(
  context,
  arm,
  select_features = NULL,
  prepend_arm_vector = FALSE
)
}
\arguments{
\item{context}{a context list containing a d x k Matrix or
d dimensional context vector X, the number of features d and
number of arms k.}

\item{arm}{index of arm.}

\item{select_features}{indices of to be returned features.}

\item{prepend_arm_vector}{prepend a one-hot-encoded arm vector to the returned context vector. That is,
when k = 5 arms, and the to be returned arm vector is arm 3, prepend c(0,0,1,0,0)}
}
\value{
Vector that represents context related to an arm
}
\description{
Given d x k matrix or d dimensional vector X,
returns a vector with arm's context.
}
