---
title: Redraw API
emoji: 🚦
colorFrom: pink
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Traffic sim API for the Redraw neighborhood planning game
---

# Redraw API

Backend for [Redraw](https://github.com/YashPatil321/REdraw): a 3D traffic simulation of
4S Ranch and Del Sur (San Diego) where players design and compare plans for the
school drop-off rush. The web client calls this API to check, save, run and vote on plans.

Built from open data (OpenStreetMap/Overture, USGS, US Census ACS and LODES); see ATTRIBUTION.md.
Saved plans live in MongoDB Atlas (`DATABASE_URL` Space secret).
