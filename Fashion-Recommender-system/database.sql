-- Create tables for the application

-- Account table
CREATE TABLE Account (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    password VARCHAR(255) NOT NULL,
    name VARCHAR(100),
    phone VARCHAR(20)
);

-- Person table
CREATE TABLE Person (
    id SERIAL PRIMARY KEY,
    preference INT REFERENCES Preference(id),
    image_directory VARCHAR(255),
    age INT,
    sex INT,
    location VARCHAR(255),
    occupation VARCHAR(100)
);

-- Preference table
CREATE TABLE Preference (
    id SERIAL PRIMARY KEY,
    brand INT REFERENCES Brand(id),
    kind INT REFERENCES Kinds(id),
    color INT REFERENCES Colors(id),
    style INT REFERENCES Styles(id)
);

-- Brand table
CREATE TABLE Brand (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);

-- Kinds table
CREATE TABLE Kinds (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);

-- Colors table
CREATE TABLE Colors (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL
);

-- Styles table
CREATE TABLE Styles (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);

-- Materials table
CREATE TABLE Materials (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL
);

-- Price Ranges table
CREATE TABLE PriceRanges (
    id SERIAL PRIMARY KEY,
    range VARCHAR(100) UNIQUE NOT NULL
);

-- Tiers table
CREATE TABLE Tiers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL -- e.g., Fast fashion, Mid, High-end
);
