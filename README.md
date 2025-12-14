# Portfolio - Hugo Sevilla Martínez
> *Status: Active Development 🟢*

This repository contains the source code for my personal engineering portfolio website (`eugegeuge.com`). The project serves as a central hub to showcase my work in **Robotics**, **Artificial Intelligence**, and **Full-Step Development**.

## Project Overview

The website is designed with a **Cyberpunk/Sci-Fi** aesthetic, reflecting a focus on futuristic technology (Digital Twins, AI, VR). It is built using standard web technologies (**HTML5**, **Tailwind CSS**, **Vanilla JavaScript**) to ensure high performance and zero-build-time complexity, deployed via **Cloudflare Pages**.

## Key Modules

### 1. Main Hub (`index.html`)
The landing page containing:
*   **Hero Section**: Introduction and specialized tags (Robotics/AI).
*   **Timeline**: A career journey visualization using a vertical timeline layout.
*   **Skills Graph**: A visual representation of technical competencies.
*   **Project Cards**: Quick links to the sub-projects detailed below.

### 2. Virtual Kinova Interface (`kinova.html`)
A dedicated landing page for the **VR Teleoperation** project.
*   **Purpose**: Demonstrates a "Digital Twin" system where a user controls a Kinova Gen2 robot using a Meta Quest 2 headset.
*   **Tech Highlight**: Explains the architecture involving Unity (VR), Python (TCP Bridge), and ROS (Robot Operating System).
*   **External Links**: Links to the valid GitHub repository for the ROS package (`Virtual-Kinova-Interface`).

### 3. MathSolver AI (`mathsolver.html`)
A showcase page for the **Computer Vision** equation solver.
*   **Purpose**: An AI system that reads handwritten equations and solves them in real-time.
*   **Tech Highlight**: Details the Convolutional Neural Network (CNN) architecture and image processing pipeline (OpenCV).

### 4. Interactive Terminal (`terminal.html`)
An easter egg feature simulating a retro Linux shell.
*   **Functionality**: Users can type commands like `help`, `cat about.txt`, `ls projects`, or `matrix` to explore the portfolio via text interface.
*   **Implementation**: A virtual file system implemented in JavaScript (JSON-based) with autocomplete and history features.

## Directory Structure

*   `assets/`: Static resources (images, icons).
*   `context/`: Contains documentation and raw files for specific sub-projects (e.g., `kinova/`, `mathsolver/`).
*   `404.html`: Custom error page with a glitch effect and hidden link to the terminal.
*   `sitemap.xml` & `robots.txt`: SEO configuration files.

## Deployment

The project is configured for **Continuous Deployment**. Any commit to the `main` branch is automatically built and published by Cloudflare.

---
*Maintained by Hugo Sevilla Martínez (2025)*