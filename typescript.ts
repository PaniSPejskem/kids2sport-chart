const extrapolatedFunction = (
    x: number,
    intervals: number[],
    values: number[]
  ): number => {
    if (intervals.length !== values.length) {
      throw new Error("Intervals and values arrays must be of the same length.");
    }
  
    if (intervals.length < 2) {
      throw new Error("At least two intervals are required for interpolation.");
    }
  
    // Below minimum: linear extrapolation
    if (x <= intervals[0]) {
      return linearInterpolation(x, intervals[0], intervals[1], values[0], values[1]);
    }
  
    // Above maximum: linear extrapolation
    if (x >= intervals[intervals.length - 1]) {
      const last = intervals.length - 1;
      return linearInterpolation(x, intervals[last - 1], intervals[last], values[last - 1], values[last]);
    }
  
    // Find the correct interval for interpolation
    for (let i = 0; i < intervals.length - 1; i++) {
      const x0 = intervals[i];
      const x1 = intervals[i + 1];
  
      if (x0 <= x && x <= x1) {
        const y0 = values[i];
        const y1 = values[i + 1];
        return linearInterpolation(x, x0, x1, y0, y1);
      }
    }
  
    // Fallback (should never reach here if data is correct)
    throw new Error("Unexpected error in extrapolatedFunction: input x out of bounds.");
  };
  
  /**
   * Performs linear interpolation or extrapolation between two points (x0, y0) and (x1, y1).
   *
   * @param x - The target x to estimate the value for
   * @param x0 - Known lower bound x
   * @param x1 - Known upper bound x
   * @param y0 - Value at x0
   * @param y1 - Value at x1
   * @returns Interpolated or extrapolated y value at x
   */
  const linearInterpolation = (
    x: number,
    x0: number,
    x1: number,
    y0: number,
    y1: number
  ): number => {
    if (x1 === x0) {
      throw new Error("Cannot interpolate with identical x values.");
    }
    const slope = (y1 - y0) / (x1 - x0);
    return y0 + slope * (x - x0);
  };
  