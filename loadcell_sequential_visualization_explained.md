# Line-by-line report: `loadcell_sequential_visualization.py`

This report follows the source file's current line numbers (1–595). Every nonblank source line is addressed individually; blank lines are identified as visual separators. The code depends on `loadcell_planned_pipeline_adc.py` and `loadcell_health_trend_pipeline.py`, whose implementation is not repeated here.

## Lines 1–32 — imports and defaults

| Line | Explanation |
|---:|---|
| 1 | Imports `argparse` to parse command-line options. |
| 2 | Imports timing functions used to measure inference latency. |
| 3 | Imports `Path` for filesystem paths. |
| 4 | Blank line separating standard-library imports. |
| 5 | Imports Matplotlib plotting functions. |
| 6 | Imports NumPy array and numeric helpers. |
| 7 | Imports `Line2D` to construct custom legend symbols. |
| 8 | Imports `FuncFormatter` to customize ADC tick labels. |
| 9 | Imports mean absolute and mean squared error metrics. |
| 10 | Imports min-max scaling for model input values. |
| 11 | Imports TensorFlow, primarily for the callback base class. |
| 12 | Imports the Keras `Model` type used to make an encoder view of the trained model. |
| 13 | Blank line separating external imports from project imports. |
| 14 | Begins a parenthesized import list from the ADC pipeline module. |
| 15 | Imports its training batch size. |
| 16 | Imports its default training epoch count. |
| 17 | Imports the temporal window length. |
| 18 | Imports the function that constructs the recurrent denoising autoencoder. |
| 19 | Imports the sliding-window sequence builder. |
| 20 | Imports fuzzy c-means clustering implementation. |
| 21 | Imports ADC file loading and preprocessing. |
| 22 | Imports overlap-aware reconstruction of a signal from windows. |
| 23 | Imports the chronological train/validation/test splitter. |
| 24 | Closes the imported-name list. |
| 25 | Imports the smoothing-window setting and health-trend calculation. |
| 26 | Blank line before configuration constants. |
| 27 | Blank line separates import area from configuration. |
| 28 | Sets the default ADC input file path. |
| 29 | Sets the default directory for generated visualizations and results. |
| 30 | Defines the nominal 535-gram reference used only for a provisional ratio display. |
| 31 | Blank line before helper functions. |
| 32 | Blank line acts as a section separator. |

## Lines 33–95 — plotting helpers and stage saving

| Line | Explanation |
|---:|---|
| 33 | Defines the helper that formats a plot axis for ADC values. |
| 34 | Displays y-axis ticks in millions of counts with two decimal places. |
| 35 | Labels that axis to make the scaling clear. |
| 36 | Blank line between functions. |
| 37 | Blank section separator. |
| 38 | Defines a reusable axis styling helper with optional axis labels. |
| 39 | Sets the plot title. |
| 40 | Checks whether an x-axis label was supplied. |
| 41 | Applies the x-axis label when present. |
| 42 | Checks whether a y-axis label was supplied. |
| 43 | Applies the y-axis label when present. |
| 44 | Adds a light grid to aid reading values. |
| 45 | Blank line between helpers. |
| 46 | Blank section separator. |
| 47 | Defines an optional progressive line animation routine. |
| 48 | Exits immediately when display/animation is disabled. |
| 49 | Opens the figure without blocking the program. |
| 50 | Begins preparing each line for progressive drawing. |
| 51 | Iterates through the supplied line objects. |
| 52 | Clears each line's current x/y data. |
| 53 | Finds the longest series to determine animation extent. |
| 54 | Chooses a chunk size that produces about 40 updates, with a minimum of one point. |
| 55 | Iterates through successive visible endpoints. |
| 56 | Pairs each line with its corresponding x and y data. |
| 57 | Limits the visible endpoint to the length of that line's data. |
| 58 | Sets the line to show only the prefix accumulated so far. |
| 59 | Recomputes axis data limits after the update. |
| 60 | Updates the displayed axis range from those limits. |
| 61 | Requests a redraw without forcing an immediate full render. |
| 62 | Processes pending GUI drawing events. |
| 63 | Pauses briefly so the animation can be seen. |
| 64 | Blank line between helpers. |
| 65 | Blank section separator. |
| 66 | Defines a progressive image/heatmap animation helper. |
| 67 | Skips animation if display is disabled. |
| 68 | Opens the plot non-blockingly. |
| 69 | Blank line separating setup steps. |
| 70 | Creates an initially empty, NaN-filled image array with the same shape as the data. |
| 71 | Displays that empty image. |
| 72 | Chooses a row increment for roughly 30 animation updates. |
| 73 | Iterates through increasing row endpoints. |
| 74 | Copies newly visible rows, capped at the total row count. |
| 75 | Updates the image artist with the current visible data. |
| 76 | Requests a redraw. |
| 77 | Processes GUI events. |
| 78 | Pauses briefly between image updates. |
| 79 | Blank line between helpers. |
| 80 | Blank section separator. |
| 81 | Defines a common function to save and optionally show one pipeline stage. |
| 82 | Builds a numbered PNG path from the stage number and name. |
| 83 | Adjusts subplot spacing before saving. |
| 84 | Saves the figure at 220 dots per inch. |
| 85 | Starts the optional interactive display branch. |
| 86 | Remembers whether Matplotlib interactive mode was enabled. |
| 87 | Turns interactive mode off for a blocking review window. |
| 88 | Draws the canvas before showing it. |
| 89 | Requests that the figure window be displayed. |
| 90 | Waits until the user closes the window. |
| 91 | Restores interactive mode only if it was originally active. |
| 92 | Re-enables interactive mode in that case. |
| 93 | Closes the figure to release resources. |
| 94 | Prints the stage name and saved image path. |
| 95 | Returns the path of the saved image. |

## Lines 96–158 — ADC/weight view and split visualization

| Line | Explanation |
|---:|---|
| 96 | Blank line before conversion helper. |
| 97 | Blank separator. |
| 98 | Defines a ratio-based ADC-to-weight display calculation. |
| 99 | Docstring warns that the result is provisional, not a calibrated measurement. |
| 100 | Uses the median ADC value as the reference level. |
| 101 | Scales each ADC reading relative to that median and the nominal load. |
| 102 | Blank line between functions. |
| 103 | Blank separator. |
| 104 | Defines the first visualization stage for raw ADC and provisional weight. |
| 105 | Computes the provisional weight series. |
| 106 | Creates two vertically stacked plots sharing the sample-index axis. |
| 107 | Blank line separates figure setup from plotting. |
| 108 | Plots raw ADC samples in a muted color. |
| 109 | Formats the ADC axis in millions of counts. |
| 110 | Titles and labels the raw ADC subplot. |
| 111 | Starts adding a note inside the raw plot. |
| 112 | Places the note near the lower-left corner in axes-relative coordinates. |
| 113 | Supplies its vertical position. |
| 114 | States that the input is raw ADC counts shown in millions. |
| 115 | Positions the note relative to the axes rather than data coordinates. |
| 116 | Gives the note a mostly opaque white background. |
| 117 | Closes the text call. |
| 118 | Blank line before the weight subplot. |
| 119 | Plots the provisional weight estimate. |
| 120 | Adds a dashed 535 g reference line and legend label. |
| 121 | Starts styling the second subplot. |
| 122 | Sets its descriptive title. |
| 123 | Supplies the x-axis label. |
| 124 | Supplies the provisional weight y-axis label and unit. |
| 125 | Closes the styling call. |
| 126 | Adds a legend for the reference line. |
| 127 | Blank line before the explanatory note. |
| 128 | Begins the note about the ratio interpretation. |
| 129 | Sets its axes-relative horizontal position. |
| 130 | Sets its vertical position. |
| 131 | Gives the formula and warns that no calibration curve exists. |
| 132 | Anchors the note to the axes coordinate system. |
| 133 | Sets the note background. |
| 134 | Closes the note call. |
| 135 | Animates the raw ADC line when `show` is true. |
| 136 | Animates only the provisional-weight data line, excluding the horizontal reference line. |
| 137 | Saves/displays and returns stage 1. |
| 138 | Blank line between stages. |
| 139 | Blank separator. |
| 140 | Defines the visualization of chronological data and window splits. |
| 141 | Unpacks train, validation, and test sample counts. |
| 142 | Creates a single wide figure. |
| 143 | Plots the entire raw ADC signal. |
| 144 | Shades the training segment and labels its sample count. |
| 145 | Shades the validation segment and labels its sample count. |
| 146 | Shades the test segment and labels its sample count. |
| 147 | Formats the ADC y-axis. |
| 148 | Sets title and sample-index label. |
| 149 | Adds a legend in the upper-right corner. |
| 150 | Begins an annotation with window-count details. |
| 151 | Sets the annotation's horizontal position. |
| 152 | Sets its vertical position. |
| 153 | Displays window size and the number of windows in each split. |
| 154 | Anchors the annotation in axes coordinates. |
| 155 | Sets an opaque note background. |
| 156 | Ends the annotation call. |
| 157 | Animates the raw signal line. |
| 158 | Saves/displays and returns stage 2. |

## Lines 159–231 — filtering reference and normalized windows

| Line | Explanation |
|---:|---|
| 159 | Blank line between stages. |
| 160 | Blank separator. |
| 161 | Defines the raw-versus-reference filtering visualization. |
| 162 | Calculates residual noise as raw input minus supplied clean reference. |
| 163 | Creates raw/reference and residual subplots. |
| 164 | Limits the displayed preview to at most 1,000 samples. |
| 165 | Plots the raw samples in the preview with partial transparency. |
| 166 | Plots the supplied moving-average reference for comparison. |
| 167 | Formats ADC values in millions. |
| 168 | Titles and labels the top subplot. |
| 169 | Adds its legend. |
| 170 | Blank line before residual plot. |
| 171 | Plots the residual over the same preview span. |
| 172 | Draws a zero residual reference line. |
| 173 | Titles and labels the residual plot. |
| 174 | Begins a statistics annotation. |
| 175 | Sets the annotation's horizontal position. |
| 176 | Sets its vertical position. |
| 177 | Reports absolute residual mean and standard deviation, and clarifies that the reference is pseudo-clean rather than ground truth. |
| 178 | Anchors annotation positioning to the axes. |
| 179 | Sets its white background. |
| 180 | Closes the annotation call. |
| 181 | Animates raw and reference lines together. |
| 182 | Animates the residual line. |
| 183 | Saves/displays and returns stage 3. |
| 184 | Blank line between stages. |
| 185 | Blank separator. |
| 186 | Defines normalization and window preparation. |
| 187 | Creates a scaler mapping values into the interval [0, 1]. |
| 188 | Fits it to the combined noisy and reference values, each reshaped as a single feature. |
| 189 | Splits noisy and clean signals chronologically into train, validation, and test segments. |
| 190 | Blank line before a local helper. |
| 191 | Defines a local function for normalizing a one-dimensional segment. |
| 192 | Reshapes values for the scaler, transforms them, then flattens them back to one dimension. |
| 193 | Blank line before window construction. |
| 194 | Starts a tuple holding all six noisy/target window arrays. |
| 195 | Creates training input windows from normalized noisy training values. |
| 196 | Creates training target windows from normalized reference values. |
| 197 | Creates validation input windows. |
| 198 | Creates validation target windows. |
| 199 | Creates test input windows. |
| 200 | Creates test target windows. |
| 201 | Closes the tuple. |
| 202 | Returns both the fitted scaler and the six window arrays. |
| 203 | Blank line between functions. |
| 204 | Blank separator. |
| 205 | Defines the plot of representative normalized windows. |
| 206 | Extracts train and test input/target windows; ignores validation arrays in this view. |
| 207 | Creates a two-by-two panel figure. |
| 208 | Plots the first training input window. |
| 209 | Adds its matching target to the same subplot. |
| 210 | Labels the training example and normalized scale. |
| 211 | Shows the training subplot legend. |
| 212 | Blank line before the window image. |
| 213 | Displays up to 80 training windows as a time-step-by-window heatmap and keeps the image artist for animation. |
| 214 | Labels the heatmap axes. |
| 215 | Blank line before test example. |
| 216 | Plots the first normalized test input. |
| 217 | Adds the matching normalized test target. |
| 218 | Labels the test example. |
| 219 | Shows the test subplot legend. |
| 220 | Blank line before explanatory panel. |
| 221 | Hides axes for the text-only panel. |
| 222 | Begins placing explanatory text in that panel. |
| 223 | Sets the horizontal position. |
| 224 | Sets the vertical position. |
| 225 | Reports the data/model shape and purpose of normalization. |
| 226 | Sets text size. |
| 227 | Aligns the text from its top edge. |
| 228 | Sets a light blue text-box background. |
| 229 | Closes the text call. |
| 230 | Animates the heatmap row-by-row when display is enabled. |
| 231 | Saves/displays and returns stage 4. |

## Lines 232–329 — training callback and encoding latency

| Line | Explanation |
|---:|---|
| 232 | Blank line between stages. |
| 233 | Blank separator. |
| 234 | Defines a Keras callback that can update a live training plot. |
| 235 | Initializes the callback with output directory and display flag. |
| 236 | Initializes the parent Keras callback. |
| 237 | Stores where the training image should be saved. |
| 238 | Stores whether interactive display is enabled. |
| 239 | Initializes a placeholder for the Matplotlib figure. |
| 240 | Initializes a placeholder for its axis. |
| 241 | Creates a list to record training losses. |
| 242 | Creates a list to record validation losses. |
| 243 | Blank line before callback hooks. |
| 244 | Defines the callback invoked at the start of model training. |
| 245 | Checks whether interactive display is off. |
| 246 | Returns without opening a plot in headless mode. |
| 247 | Creates the live loss figure and axis. |
| 248 | Sets the GUI window title. |
| 249 | Enables Matplotlib interactive mode. |
| 250 | Blank line between hooks. |
| 251 | Defines the callback invoked after each training epoch. |
| 252 | Uses an empty dictionary if Keras supplies no logs. |
| 253 | Appends the epoch training loss, using NaN if unavailable. |
| 254 | Appends validation loss, likewise defaulting to NaN. |
| 255 | Checks for headless mode. |
| 256 | Returns after recording metrics if no live plot is requested. |
| 257 | Clears the axis so it can be redrawn. |
| 258 | Plots recorded training MSE values. |
| 259 | Plots recorded validation MSE values. |
| 260 | Updates the title with current and total epoch numbers. |
| 261 | Labels the horizontal axis. |
| 262 | Labels the loss axis. |
| 263 | Adds a legend. |
| 264 | Adds a light grid. |
| 265 | Adjusts figure layout. |
| 266 | Requests a redraw. |
| 267 | Processes GUI events. |
| 268 | Pauses briefly to show the update. |
| 269 | Blank line between hooks. |
| 270 | Defines the callback invoked after training completes. |
| 271 | Skips interactive cleanup in headless mode. |
| 272 | Returns if no plot window was created. |
| 273 | Adjusts layout before saving. |
| 274 | Saves the training curve as `05_rdae_training.png`. |
| 275 | Records whether interactive mode was on. |
| 276 | Turns interactive mode off for the blocking review. |
| 277 | Displays the training figure. |
| 278 | Waits for the plot window to close. |
| 279 | Restores interactive mode if it was previously active. |
| 280 | Re-enables interactive mode as needed. |
| 281 | Closes the training figure. |
| 282 | Blank line between class and next function. |
| 283 | Blank separator. |
| 284 | Defines test-set bottleneck encoding and latency visualization. |
| 285 | Creates a model that shares the trained model input and outputs the `bottleneck` layer. |
| 286 | Computes the number of batches needed, rounding up. |
| 287 | Creates latency and latent-feature subplots. |
| 288 | Creates one zero-height bar per batch for live latency updates. |
| 289 | Labels the latency chart. |
| 290 | Creates the first latent-feature line. |
| 291 | Adds the second latent-feature line. |
| 292 | Labels the bottleneck output chart. |
| 293 | Displays its legend. |
| 294 | Starts optional live display. |
| 295 | Opens the figure without blocking. |
| 296 | Blank line before inference measurement. |
| 297 | Initializes a list of per-batch elapsed times. |
| 298 | Initializes a list to hold encoded outputs. |
| 299 | Iterates through test inputs in batches of `BATCH_SIZE`. |
| 300 | Slices the current batch from test windows. |
| 301 | Records a high-resolution start timestamp. |
| 302 | Runs encoder inference without Keras progress output. |
| 303 | Computes elapsed inference milliseconds. |
| 304 | Records that batch duration. |
| 305 | Stores that batch's encoded vectors. |
| 306 | Starts the live-update branch. |
| 307 | Gets the zero-based index of the just-completed batch. |
| 308 | Updates that batch's latency bar. |
| 309 | Concatenates all encoded batches processed so far. |
| 310 | Builds sequential x coordinates for encoded windows. |
| 311 | Updates the first latent-feature curve. |
| 312 | Updates the second latent-feature curve. |
| 313 | Recalculates latent plot limits. |
| 314 | Applies the new view limits. |
| 315 | Updates the figure title with progress. |
| 316 | Requests canvas redraw. |
| 317 | Processes GUI events. |
| 318 | Pauses so the live update is visible. |
| 319 | Blank line before aggregation. |
| 320 | Concatenates all encoded batches into the final latent matrix. |
| 321 | Sums batch durations into total encoding time. |
| 322 | Calculates average latency per test window. |
| 323 | Starts a final summary title. |
| 324 | Reports total and per-window time, plus bottleneck dimension. |
| 325 | Sets title font size. |
| 326 | Closes the title call. |
| 327 | Packages latent vectors and latency measurements into a dictionary. |
| 328 | Saves/shows the stage 6 figure. |
| 329 | Returns the result dictionary. |

## Lines 330–421 — reconstruction, metrics, and output plot

| Line | Explanation |
|---:|---|
| 330 | Blank line between stages. |
| 331 | Blank separator. |
| 332 | Defines model prediction, signal reconstruction, and evaluation plot. |
| 333 | Reshapes test windows into the 3D input shape expected by Keras. |
| 334 | Calculates the original signal length represented by overlapping windows. |
| 335 | Reconstructs one continuous target signal from target windows. |
| 336 | Reconstructs the continuous noisy input from input windows. |
| 337 | Converts normalized target values back to ADC units and flattens them. |
| 338 | Blank line separating inverse transforms. |
| 339 | Converts the reconstructed noisy input back to ADC units. |
| 340 | Limits the plotted preview while leaving room for edge effects. |
| 341 | Creates two vertically stacked panels. |
| 342 | Defines x coordinates that align with the center-valid portion of each window reconstruction. |
| 343 | Plots the noisy test input, excluding half a window at the start. |
| 344 | Plots the pseudo-clean target over the same sample positions. |
| 345 | Creates an initially empty line for model output animation. |
| 346 | Formats the top panel's ADC ticks. |
| 347 | Labels the live filtering chart. |
| 348 | Adds its legend. |
| 349 | Starts optional live display. |
| 350 | Opens the plot non-blockingly. |
| 351 | Blank line before prediction. |
| 352 | Creates a list to collect prediction batches. |
| 353 | Iterates over test windows in batches. |
| 354 | Starts clipping model predictions into the normalized range. |
| 355 | Predicts the current batch without console progress output. |
| 356 | Sets lower allowed normalized prediction. |
| 357 | Sets upper allowed normalized prediction. |
| 358 | Closes the clipping call. |
| 359 | Stores the current predicted batch. |
| 360 | Starts optional live plot updates. |
| 361 | Combines predictions so far. |
| 362 | Computes the partial reconstructed signal length. |
| 363 | Overlap-averages partial windows into a signal. |
| 364 | Converts partial reconstruction back to ADC units. |
| 365 | Limits its visible extent to the preview size. |
| 366 | Begins updating the output line. |
| 367 | Sets x coordinates aligned to the center-valid segment. |
| 368 | Supplies that segment's reconstructed ADC values. |
| 369 | Ends the line update. |
| 370 | Recomputes plot limits. |
| 371 | Updates the plot view. |
| 372 | Starts a progress title update. |
| 373 | Reports completed prediction windows. |
| 374 | Sets title font size. |
| 375 | Closes the title call. |
| 376 | Requests a redraw. |
| 377 | Processes GUI events. |
| 378 | Pauses to expose the update. |
| 379 | Blank line before final metrics. |
| 380 | Combines all prediction batches. |
| 381 | Reconstructs the full normalized output, then clips it to [0, 1]. |
| 382 | Converts the output into ADC units. |
| 383 | Defines the start index after the first half-window edge region. |
| 384 | Defines the matching end index before the trailing edge region. |
| 385 | Measures noisy-input MSE against the pseudo-clean target on valid indices. |
| 386 | Measures noisy-input MAE over the same indices. |
| 387 | Measures reconstructed-output MSE against the target. |
| 388 | Measures reconstructed-output MAE against the target. |
| 389 | Blank line before final curve update. |
| 390 | Sets the output curve to the reconstructed valid preview. |
| 391 | Recalculates axis limits. |
| 392 | Applies the recalculated view. |
| 393 | Blank line before text-only panel. |
| 394 | Hides the lower panel axes. |
| 395 | Begins explanatory metrics text. |
| 396 | Places the text near the upper-left. |
| 397 | Sets its vertical position. |
| 398 | Explains intended output and reports noisy/output errors, explicitly noting the pseudo-clean target limitation. |
| 399 | Sets font size. |
| 400 | Aligns text to the top. |
| 401 | Gives the note a light green background. |
| 402 | Closes the text call. |
| 403 | Starts animating all three signal lines. |
| 404 | Supplies the figure object. |
| 405 | Supplies the signal axis. |
| 406 | Supplies noisy, target, and reconstructed line artists. |
| 407 | Supplies their x coordinate arrays. |
| 408 | Supplies the corresponding valid y data arrays. |
| 409 | Passes through whether animation is enabled. |
| 410 | Closes the animation call. |
| 411 | Saves/displays stage 7. |
| 412 | Begins the returned results mapping. |
| 413 | Includes normalized prediction windows. |
| 414 | Includes the continuous reconstructed ADC signal. |
| 415 | Includes the continuous pseudo-clean target. |
| 416 | Includes the continuous noisy input. |
| 417 | Includes noisy-input MSE. |
| 418 | Includes noisy-input MAE. |
| 419 | Includes reconstructed-output MSE. |
| 420 | Includes reconstructed-output MAE. |
| 421 | Closes and returns the dictionary. |

## Lines 422–476 — clustering and health trend

| Line | Explanation |
|---:|---|
| 422 | Blank line between stages. |
| 423 | Blank separator. |
| 424 | Defines a scatter plot for fuzzy-clustering features. |
| 425 | Selects the third feature from the end as mean reconstruction error. |
| 426 | Selects the second feature from the end as standard deviation of error. |
| 427 | Creates the scatter plot figure. |
| 428 | Plots each window's two error features, colored by cluster label. |
| 429 | Plots cluster centers as red X symbols using the corresponding feature columns. |
| 430 | Starts a list of custom legend handles. |
| 431 | Configures a circular marker using the scatter plot's cluster color mapping. |
| 432 | Removes marker edge and assigns marker size and cluster label. |
| 433 | Repeats that legend marker for each distinct cluster ID. |
| 434 | Closes the list comprehension. |
| 435 | Adds a red X legend entry for the centroid markers. |
| 436 | Displays the legend with the title “FCM output.” |
| 437 | Labels the cluster plot and its error axes. |
| 438 | Begins optional animation branch. |
| 439 | Opens the plot non-blockingly. |
| 440 | Clears all scatter points before animation. |
| 441 | Calculates a point increment and iterates through progress endpoints. |
| 442 | Adds the first `end` error-feature points to the scatter. |
| 443 | Requests a redraw. |
| 444 | Processes GUI events. |
| 445 | Pauses briefly between batches of points. |
| 446 | Adds all cluster centers after the observations appear. |
| 447 | Requests a redraw. |
| 448 | Processes GUI events. |
| 449 | Pauses briefly to show the centroids. |
| 450 | Saves/displays and returns stage 8. |
| 451 | Blank line between stages. |
| 452 | Blank separator. |
| 453 | Defines the health trend and state visualization. |
| 454 | Computes health trend arrays and summary statistics using configured smoothing. |
| 455 | Creates one x position per test window. |
| 456 | Creates two vertically stacked plots sharing x. |
| 457 | Plots fuzzy cluster severity. |
| 458 | Plots severity based on reconstruction error. |
| 459 | Plots the smoothed combined degradation index. |
| 460 | Sets a title containing slope and monotonicity statistics and labels axes. |
| 461 | Fixes severity display to approximately the normalized range [0, 1]. |
| 462 | Adds a legend for the severity curves. |
| 463 | Plots health state IDs as a step series. |
| 464 | Maps state IDs 0, 1, and 2 to Healthy, Warning, and Degraded labels. |
| 465 | Titles and labels the health-state subplot. |
| 466 | Begins animation of the three upper-panel lines. |
| 467 | Supplies the figure. |
| 468 | Supplies the severity axis. |
| 469 | Supplies its line artists. |
| 470 | Supplies a copy of window indices for each line. |
| 471 | Supplies fuzzy, reconstruction, and smoothed severity data. |
| 472 | Passes the show flag. |
| 473 | Closes the animation call. |
| 474 | Animates the state step line. |
| 475 | Saves/displays stage 9. |
| 476 | Returns the computed trend mapping. |

## Lines 477–504 — final summary image

| Line | Explanation |
|---:|---|
| 477 | Blank line between stages. |
| 478 | Blank separator. |
| 479 | Defines a final report figure summarizing pipeline outputs. |
| 480 | Creates the figure and one axis. |
| 481 | Hides axes because the panel contains text only. |
| 482 | Counts windows in each of the three health states, ensuring three bins. |
| 483 | Starts assembling the summary text. |
| 484 | Adds the report heading. |
| 485 | Adds the source filename. |
| 486 | Adds the ADC sample count with thousands separators. |
| 487 | Adds window counts for the train, validation, and test splits. |
| 488 | Adds total encoder inference latency. |
| 489 | Adds encoder latency per test window. |
| 490 | Adds baseline noisy test MAE. |
| 491 | Adds denoised model output MAE. |
| 492 | Adds health trend slope. |
| 493 | Adds trend monotonicity. |
| 494 | Adds counts for the Healthy, Warning, and Degraded states. |
| 495 | Adds a short expected-output heading. |
| 496 | Lists the denoised/reconstructed ADC test signal. |
| 497 | Lists bottleneck features and fuzzy clusters. |
| 498 | Lists the health/degradation indicator and its later-validation context. |
| 499 | Notes that generated files are listed below. |
| 500 | Ends the composed summary string. |
| 501 | Renders the summary in monospaced text. |
| 502 | Collects paths of PNG files currently in the output directory. |
| 503 | Displays those paths in the bottom of the figure. |
| 504 | Saves/displays and returns stage 10. |

## Lines 505–582 — end-to-end pipeline execution

| Line | Explanation |
|---:|---|
| 505 | Blank line before orchestration function. |
| 506 | Blank separator. |
| 507 | Defines the main pipeline runner with input, output, epochs, and display settings. |
| 508 | Creates the output directory, including missing parent folders. |
| 509 | Starts interactive plotting setup when display is enabled. |
| 510 | Enables Matplotlib interactive mode. |
| 511 | Blank line before loading data. |
| 512 | Loads sample indices, reference signal, and noisy ADC signal from the file. |
| 513 | Produces the initial ADC and relative-weight plot. |
| 514 | Blank line separates initial visualization from data splitting. |
| 515 | Splits noisy and clean/reference series into chronological partitions. |
| 516 | Records sample counts for each partition. |
| 517 | Fits normalization and creates sliding windows. |
| 518 | Unpacks the six arrays returned for train, validation, and test. |
| 519 | Records the number of windows in each split. |
| 520 | Creates the split overview visualization. |
| 521 | Creates the filtering/reference visualization. |
| 522 | Creates normalized-window visualization. |
| 523 | Blank line before model creation. |
| 524 | Builds the recurrent denoising autoencoder for the configured window length. |
| 525 | Creates the loss-recording/live-plot callback. |
| 526 | Starts Keras model training. |
| 527 | Reshapes noisy training windows to `[windows, time steps, 1 feature]`. |
| 528 | Reshapes target training windows to the same 3D layout. |
| 529 | Supplies reshaped validation input and target arrays. |
| 530 | Sets the number of training epochs. |
| 531 | Sets batch size. |
| 532 | Suppresses Keras's standard progress display. |
| 533 | Registers the live training callback. |
| 534 | Closes the fit call. |
| 535 | Starts the headless-only plot saving branch. |
| 536 | Creates a figure for the recorded loss curves. |
| 537 | Plots training loss collected by the callback. |
| 538 | Plots validation loss. |
| 539 | Titles and labels the training result chart. |
| 540 | Adds its legend. |
| 541 | Saves stage 5 without opening a window. |
| 542 | Blank line before test encoding. |
| 543 | Reshapes test input to model-compatible 3D form. |
| 544 | Encodes test windows and gathers latency statistics. |
| 545 | Extracts latent/bottleneck vectors from the latency result. |
| 546 | Predicts reconstructed normalized windows and clips them to [0, 1]. |
| 547 | Reshapes test targets to model output shape. |
| 548 | Calculates absolute per-value prediction error. |
| 549 | Begins building features for clustering. |
| 550 | Includes bottleneck features as the first columns. |
| 551 | Appends each window's mean absolute error. |
| 552 | Appends each window's error standard deviation. |
| 553 | Appends each window's maximum absolute error. |
| 554 | Concatenates feature columns horizontally. |
| 555 | Runs three-cluster fuzzy c-means and receives centers, memberships, labels, and an additional result. |
| 556 | Plots reconstructed output and computes its metrics. |
| 557 | Plots clusters and centers. |
| 558 | Builds and plots the health trend. |
| 559 | Blank line before CSV export. |
| 560 | Sets the per-window CSV output path. |
| 561 | Starts writing a numeric CSV using NumPy. |
| 562 | Supplies the destination path. |
| 563 | Begins assembling the CSV columns. |
| 564 | Adds zero-based test-window index. |
| 565 | Adds cluster label. |
| 566 | Adds fuzzy severity. |
| 567 | Adds reconstruction-error severity. |
| 568 | Adds smoothed degradation. |
| 569 | Adds encoded health-state ID. |
| 570 | Closes the column stack. |
| 571 | Selects comma separators. |
| 572 | Writes the CSV header names. |
| 573 | Prevents NumPy's default comment marker on the header. |
| 574 | Closes the CSV writer call. |
| 575 | Begins a small summary dictionary. |
| 576 | Stores total sample count. |
| 577 | Stores train/validation/test window counts. |
| 578 | Closes the dictionary. |
| 579 | Creates the final summary image. |
| 580 | Prints where the per-window CSV was saved. |
| 581 | Prints a completion message. |
| 582 | Blank line before command-line entry point. |

## Lines 583–595 — command-line entry point

| Line | Explanation |
|---:|---|
| 583 | Blank separator. |
| 584 | Defines command-line `main`. |
| 585 | Creates an argument parser and describes this program. |
| 586 | Adds `--file`, parsed as a path and defaulting to the configured ADC file. |
| 587 | Adds `--output-dir` with the default results directory. |
| 588 | Adds an integer epoch-count option, defaulting to the pipeline's `EPOCHS`. |
| 589 | Adds `--no-show`; when present, figures are saved without opening separate windows. |
| 590 | Parses command-line arguments. |
| 591 | Runs the pipeline, enforcing at least one epoch and converting `--no-show` to a display boolean. |
| 592 | Blank line before the script guard. |
| 593 | Blank separator. |
| 594 | Ensures `main()` runs only when this file is executed directly, not when imported. |
| 595 | Calls the command-line entry point. |

## Overall program flow

The script loads ADC data, shows the raw and provisional weight views, splits the series chronologically, normalizes it, and forms temporal windows. It trains an R-DAE, measures bottleneck encoding latency, predicts the test windows, computes reconstruction-error features, clusters the windows with fuzzy c-means, and derives a smoothed health trend. It saves ten stage images and a per-window CSV under the selected output directory.

The “weight” curve is explicitly only a ratio view based on the median ADC reading and the filename's nominal load. Likewise, the filtering metrics compare against a moving-average pseudo-clean reference, not independent calibrated ground truth.
