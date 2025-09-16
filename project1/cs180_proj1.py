<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>CS180/CS280A Project 1 — Colorizing the Prokudin-Gorskii Photo Collection</title>
  <style>
    body {font-family: Arial, sans-serif; font-size:16px; line-height:1.6; margin:0; padding:20px; background:#fff; color:#000;}
    img {max-width:100%; height:auto; display:block; margin:10px 0 0 0;}
    figure {margin:20px 0;}
    figcaption {font-size:0.9rem; color:#555; margin-top:4px;}
    table {border-collapse: collapse; width:100%; margin:20px 0;}
    th,td {border:1px solid #ccc; padding:6px 10px; text-align:left;}
    thead th {background:#f1f5f9;}
    ul {margin:10px 0 20px 20px;}
  </style>
</head>
<body>
  <h1>CS180/CS280A Project 1 — Colorizing the Prokudin-Gorskii Photo Collection</h1>

  <h2>Offsets Table</h2>
  <table>
    <thead>
      <tr>
        <th>Image</th>
        <th>B→G (dx, dy)</th>
        <th>R→G (dx, dy)</th>
        <th>Method</th>
      </tr>
    </thead>
    <tbody>
      <tr><td>emir.tif</td><td>(-24, -49)</td><td>(17, 57)</td><td>Multi-scale</td></tr>
      <tr><td>italil.tif</td><td>(-21, -38)</td><td>(15, 39)</td><td>Multi-scale</td></tr>
      <tr><td>monastery.jpg</td><td>(-2, 3)</td><td>(1, 6)</td><td>Single-scale</td></tr>
      <tr><td>church.tif</td><td>(-4, -25)</td><td>(-8, 33)</td><td>Multi-scale</td></tr>
      <tr><td>three_generations.tif</td><td>(-14, -53)</td><td>(-3, 58)</td><td>Multi-scale</td></tr>
      <tr><td>lugano.tif</td><td>(16, -41)</td><td>(-13, 52)</td><td>Multi-scale</td></tr>
      <tr><td>melons.tif</td><td>(-11, -82)</td><td>(4, 96)</td><td>Multi-scale</td></tr>
      <tr><td>lastochikino.tif</td><td>(2, 3)</td><td>(-7, 78)</td><td>Multi-scale</td></tr>
      <tr><td>tobolsk.jpg</td><td>(-3, -3)</td><td>(1, 4)</td><td>Single-scale</td></tr>
      <tr><td>icon.tif</td><td>(-17, -41)</td><td>(5, 48)</td><td>Multi-scale</td></tr>
      <tr><td>siren.tif</td><td>(6, -49)</td><td>(-18, 47)</td><td>Multi-scale</td></tr>
      <tr><td>self_portrait.tif</td><td>(-29, -79)</td><td>(8, 98)</td><td>Multi-scale</td></tr>
      <tr><td>harvesters.tif</td><td>(-17, -60)</td><td>(-3, 65)</td><td>Multi-scale</td></tr>
    </tbody>
  </table>
  <p>All offsets are integer pixel shifts that align B and R channels to G.</p>

  <h2>Results</h2>

  <h3>Single-scale Alignment Results</h3>
  <ul>
    <li>Took Green channel as reference as it normally has lower chromatic abberration according to past experience</li>
    <li>Exhaustively searched over a fixed window of (dx=40pix, dy=40pix) displacements for B and R relative to G channel image.</li>
    <li>I obtained poor results without cropping the image borders and then realized the source of noise and cropped borders(8%) to ignore artifacts.</li> 
    <li> For each displacement, SSD (Sum of squared differences) and NCC (Normalized cross correlation) were used as a metric to evaluate image alignment quality on both native rgb images as well as on edge detected image.</li>
  </ul>
  <figure>
    <img src="results/monastery.jpg" alt="monastery.jpg result">
    <figcaption>monastery.jpg — Single-scale alignment result</figcaption>
  </figure>
  <figure>
    <img src="results/tobolsk.jpg" alt="tobolsk.jpg result">
    <figcaption>tobolsk.jpg — Single-scale alignment result</figcaption>
  </figure>
  <figure>
    <img src="results/cathedral.jpg" alt="cathedral.jpg result">
    <figcaption>cathedral.jpg — Single-scale alignment result</figcaption>
  </figure>

  <h3>Multi-scale Alignment Results</h3>
  <ul>
    <li>Created an image pyramid of total levels = 4. </li>
    <li>To do this, images were first smoothed out of high frequency noise by applying Gaussian kernel. Since gaussian can be split, I have used a 1D Gaussian kernel for x and y direction (in x and y direction) of sigma=1 and width=7 </li>
    <l1> After smoothing out the image (which reduces aliasing), we remove every other row and column to downsize the image by scale = 2. We do this exerise couple of more times until pyramids of all levels are computed.
    <li>Starting at the coarsest level, I estimated image alignment with ncc as criterion.</li>
    <li>The (dx,dy) obtained between R and G and B and G is scaled by 2 for the next level (less coarse image) and used it as the starting point at the next finer level.</li>
    <li>As the levels get towards finer images, the window of the gaussian is reduced because we narrow down our belief of dx,dy</li>
  </ul>
  <figure>
    <img src="results/emir.jpg" alt="emir.tif result">
    <figcaption>emir.tif — Multi-scale alignment result</figcaption>
  </figure>
  <figure>
    <img src="results/italil.jpg" alt="italil.tif result">
    <figcaption>italil.tif — Multi-scale alignment result</figcaption>
  </figure>
  <figure>
    <img src="results/church.jpg" alt="church.tif result">
    <figcaption>church.tif — Multi-scale alignment result</figcaption>
  </figure>
  <figure>
    <img src="results/three_generations.jpg" alt="three_generations.tif result">
    <figcaption>three_generations.tif — Multi-scale alignment result</figcaption>
  </figure>
  <figure>
    <img src="results/lugano.jpg" alt="lugano.tif result">
    <figcaption>lugano.tif — Multi-scale alignment result</figcaption>
  </figure>
  <figure>
    <img src="results/melons.jpg" alt="melons.tif result">
    <figcaption>melons.tif — Multi-scale alignment result</figcaption>
  </figure>
  <figure>
    <img src="results/lastochikino.jpg" alt="lastochikino.tif result">
    <figcaption>lastochikino.tif — Multi-scale alignment result</figcaption>
  </figure>
  <figure>
    <img src="results/icon.jpg" alt="icon.tif result">
    <figcaption>icon.tif — Multi-scale alignment result</figcaption>
  </figure>
  <figure>
    <img src="results/siren.jpg" alt="siren.tif result">
    <figcaption>siren.tif — Multi-scale alignment result</figcaption>
  </figure>
  <figure>
    <img src="results/self_portrait.jpg" alt="self_portrait.tif result">
    <figcaption>self_portrait.tif — Multi-scale alignment result</figcaption>
  </figure>
  <figure>
    <img src="results/harvesters.jpg" alt="harvesters.tif result">
    <figcaption>harveste
