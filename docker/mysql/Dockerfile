FROM mysql:latest

ENV MYSQL_ROOT_PASSWORD=ta2min
ENV MYSQL_DATABASE=ta2min
ENV MYSQL_USER=user
ENV MYSQL_PASSWORD=ta2min

COPY football_players.csv /docker-entrypoint-initdb.d/

WORKDIR /docker-entrypoint-initdb.d/

RUN echo "CREATE TABLE IF NOT EXISTS football_players (id INT AUTO_INCREMENT PRIMARY KEY, Name VARCHAR(255), Age INT, Photo VARCHAR(255), Nationality VARCHAR(255), Flag VARCHAR(255), Potential INT, Club VARCHAR(255), Club_Logo VARCHAR(255), Value_Pound VARCHAR(255));" > init.sql
RUN echo "LOAD DATA INFILE '/docker-entrypoint-initdb.d/football_players.csv' INTO TABLE football_players FIELDS TERMINATED BY ',' ENCLOSED BY '\"' LINES TERMINATED BY '\\n' IGNORE 1 ROWS (@dummy, Name, Age, Photo, Nationality, Flag, @dummy, Potential, Club, Club_Logo, Value_Pound, @dummy, @dummy, @dummy, @dummy, @dummy, @dummy, @dummy, @dummy, @dummy, @dummy, @dummy, @dummy, @dummy);" >> init.sql

EXPOSE 3306
