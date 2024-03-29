# Project Introduction

In the dynamic world of sports, where team managers are tasked with building and enhancing their teams, our Recommendation System comes to the forefront as a powerful tool designed to assist team managers in making informed decisions. Whether you are a seasoned team manager or just starting out, our platform aims to streamline the player selection process by incorporating intelligent recommendations for insurance offers tailored to the preferences and historical choices of each team's unique player.

## Project Overview

### Objective

The primary objective of our Recommendation System is to empower team managers with a data-driven approach when assigning insurance offers to their players. By leveraging machine learning techniques and collaborative filtering algorithms, the system analyzes player data, preferences, and historical choices to provide top-k recommendations for insurance offers that align with the individual player's profile.

### How It Works

1. **Player Information Input:**
    - Team managers input detailed information about each player, including personal details, performance metrics, and other relevant attributes.
2. **Preferences Analysis:**
    - The system analyzes the preferences of each player, considering factors such as age, overall performance, and club affiliations.
3. **Collaborative Filtering:**
    - Leveraging collaborative filtering, the system identifies similar players based on historical data and preferences.
4. **Top-k Recommendations:**
    - The system generates top-k recommendations for insurance offers by comparing the player's preferences with those of similar players.

### Use Case Scenario

Imagine a team manager adding a new player to their roster. When it comes time to assign insurance offers, the team manager may not have extensive knowledge of the insurance market. This is where our Recommendation System steps in, providing personalized recommendations based on the player's characteristics and the choices of players with similar profiles.

## Key Features

- **Dynamic Recommendations:** The system adapts to changes in player data and preferences, ensuring recommendations stay relevant over time.
- **User-Friendly Interface:** Team managers can easily navigate the platform, input player data, and access insurance recommendations seamlessly.
- **Scalability:** The system is designed to handle growing datasets and an increasing number of users, making it suitable for teams of all sizes.

# Technologies & Tools

## Backend Development

### 1. Python Programming Language and Jupyter Notebook

- **Use Cases:**
    - Backend logic implementation.
    - Machine learning and data analysis.

### 2. Flask Framework

- **Use Case:**
    - API development for communication between frontend and backend.

### 3. Git/GitHub

- **Use Cases:**
    - Version control for collaborative development.
    - Source code management.

### 4. Apache Kafka

![Untitled](https://github.com/2aman-lafr9a/RecommendationSystem/assets/90706276/21df5958-b8a5-4a19-9ca3-8f561e0262e6)

- **Use Case:**
    - Kafka facilitates real-time event streaming, essential for dynamic updates in the recommendation system.

### 5. Apache Zookeeper

![Untitled(1)](https://github.com/2aman-lafr9a/RecommendationSystem/assets/90706276/dc3cae98-bea5-4477-8b5d-04c92750a66b)

- **Use Case:**
    - Zookeeper ensures distributed coordination and synchronization.

## MLOps

### 1. Docker

![Untitled(2)](https://github.com/2aman-lafr9a/RecommendationSystem/assets/90706276/c544d466-2f20-4b2d-95c9-46d5e6535c5d)

- **Use Case:**
    - Containerization for consistent deployment across environments.

### 2. Apache Airflow

![Untitled(3)](https://github.com/2aman-lafr9a/RecommendationSystem/assets/90706276/df79daf4-f709-4105-9207-4eed52fbad9e)

- **Use Cases:**
    - Orchestrating and scheduling workflows.
    - Automation of data processing and model training pipelines.

### 3. Data Version Control (DVC)

![file-type-dvc-icon-512x293-js3het8o](https://github.com/2aman-lafr9a/RecommendationSystem/assets/90706276/6163b22e-1155-4e82-a2fd-d4ad3d517486)

- **Use Case:**
    - Efficient versioning of datasets for reproducibility and traceability.

### 4. CI/CD using GitHub Actions (CML - Continuous Machine Learning)

![Untitled(4)](https://github.com/2aman-lafr9a/RecommendationSystem/assets/90706276/3a9b7b6a-43de-4b09-99bc-28bbdd3d70b4)

- **Use Cases:**
    - Continuous integration for automated testing.
    - Continuous deployment for seamless updates.

## Databases

### 1. Cassandra Database

![Untitled(5)](https://github.com/2aman-lafr9a/RecommendationSystem/assets/90706276/ca17e3e7-8238-412a-8158-8b2d04570e5b)

- **Use Cases:**
    - Scalable storage and retrieval of player data.
    - Efficient handling of large datasets.

### 2. PostgreSQL Database

- **Use Cases:**
    - Management of rating data and insurances data.
    - Relational database capabilities.

# Getting Started

...

# Recommendation System Overview

A **Recommendation System** is a software application designed to provide personalized suggestions or advice to users. Its primary goal is to enhance user experience by predicting and recommending items of interest, such as products, services, or content, based on user preferences and historical interactions.

## Collaborative Filtering

**Collaborative Filtering** is a popular technique in Recommendation Systems that leverages the collective preferences and behavior of users. It assumes that users who agreed in the past tend to agree in the future. There are two main types:

1. **User-User Collaborative Filtering:** Recommends items to a user based on the preferences of other users with similar tastes.
2. **Item-Item Collaborative Filtering:** Recommends items that are similar to those the user has liked or interacted with.

![Untitled-2024-01-18-1833](https://github.com/2aman-lafr9a/RecommendationSystem/assets/90706276/6664becc-0c94-48a4-9ed8-fb900d8ec125)

## User-Item Collaborative Filtering

**User-Item Collaborative Filtering** is a type of Collaborative Filtering where recommendations are made by identifying users who are similar to the target user and suggesting items that those similar users have liked or interacted with.

![RecSys1](https://github.com/2aman-lafr9a/RecommendationSystem/assets/90706276/fe058e49-e070-4969-af4b-b911b6864540)

# Architecture

## 1. **Pre-processing**

![RecSys3](https://github.com/2aman-lafr9a/RecommendationSystem/assets/90706276/8d4b0338-1c5d-42ff-be43-4ac436000e44)

## 2. Model Training

![RecSys4](https://github.com/2aman-lafr9a/RecommendationSystem/assets/90706276/5b480d9c-8cef-4fd9-af90-bce07555fb48)

## 3. User-Based Collaborative Filtering

![RecSys5](https://github.com/2aman-lafr9a/RecommendationSystem/assets/90706276/6891d9c3-e383-4e92-848b-fe5e887da321)

![RecSys6](https://github.com/2aman-lafr9a/RecommendationSystem/assets/90706276/4c2443d2-382f-47c5-a2db-beaf0ac0f236)
