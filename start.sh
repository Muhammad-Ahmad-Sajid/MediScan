#!/bin/bash
# start.sh
# Run this script to manually execute the SQL schema (database.sql) against the PostgreSQL container.

echo "Waiting for the database container to be fully ready..."
sleep 5

echo "Applying schema from database.sql to fracture_db..."
docker-compose exec -T db psql -U postgres -d fracture_db < database.sql

echo "Done! The database schema has been applied."
