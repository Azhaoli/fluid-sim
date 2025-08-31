use std::io;
use std::io::Write;
use std::time::Duration;
use std::thread::sleep;
use std::cmp::{max, min};
use std::f32::consts::PI;

#[derive(Clone)]
struct Grid {
	elems: Vec<Vec<f32>>,
	size: usize, // plot side length is positive
}

impl Grid {
	fn new(size: usize) -> Grid {
		let mut elems = Vec::new();
		for x in 0..size { elems.push(vec![0.0; size]); }  // fill the grid with 0s
		Grid{ elems, size }
	}
	
	fn bounding_box(size: usize) -> Grid {
		let mut elems = Vec::new();
		let mut column = Vec::new();
	
		for y in 0..size {
			for x in 0..size {
				if (x==0) || (y==0) || (x == size-1) || (y == size-1){ column.push(0.0); }else { column.push(1.0); }
			}
			elems.push(column.clone());
			column.clear();
		}
		Grid{ elems, size }
	}
	
	fn get(&self, x: usize, y: usize) -> f32 {
		if (x >= self.size) || (y >= self.size) { 0.0 }
		else { self.elems[x][y]	}
	}

	fn render(&self) {
		let (mut x, mut y) = (0, 0);
		let size = self.size;
		while y < size {
			while x < size {
				let upper_color = value_to_color(self.get(x, y));
				let lower_color = value_to_color(self.get(x, y+1));
			
				print!(
					"\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀\x1b[0m",
					upper_color[0], upper_color[1], upper_color[2],
					lower_color[0], lower_color[1], lower_color[2]
				);
				x += 1;
			}
			io::stdout().flush().unwrap();
			println!("");
			y += 2;
			x = 0;
		}
	}
	
	fn render_values(&self) {
		for y in 0..self.size {
			for x in 0..self.size { print!("<{:.3}>", self.get(x, y)); }
			io::stdout().flush().unwrap();
			println!("");
		}
	}
}


fn lerp(a: f32, b: f32, k: f32) -> f32 { a + k*(b-a) }

// map value to color range
fn value_to_color(num: f32) -> [usize; 3] {
	
	HSL_to_RGB(0.0, 0.0, (2.0/PI)*num.atan())
}

fn vec_to_color(x: f32, y: f32) -> [usize; 3] {
	let mag = (x*x + y*y).sqrt();
	let angle = (y/x).atan();
	// 2d vector to HSL
	let arg = if x < 0.0 { angle + 3.0*PI/2.0 }else { angle + PI/2.0 };

	let H = (arg + 2.0*PI)%(2.0*PI)* (180.0/PI);
	let S = 0.999; // saturation 0-1
	let L = (2.0/PI)*mag.atan(); // lightness/ value 0-1
	HSL_to_RGB(H, S, L)
}

fn HSL_to_RGB(H: f32, S: f32, L: f32) -> [usize; 3] {
	let H1 = H /60.0;
	let C = S * (1.0 - (2.0*L - 1.0).abs()); // find chroma
	let X = C * (1.0 - (H1%2.0 - 1.0).abs());
	
	let [R_1, G_1, B_1] = match H1 {
		0.0..=1.0 => [C, X, 0.0],
		1.0..=2.0 => [X, C, 0.0],
		2.0..=3.0 => [0.0, C, X],
		3.0..=4.0 => [0.0, X, C],
		4.0..=5.0 => [X, 0.0, C],
		5.0..=6.0 => [C, 0.0, X],
		_ => [0.0, 0.0, 0.0]
	};
	let m = L - C/2.0;
	
	[
		(255.0 * (R_1 + m)) as usize,
		(255.0 * (G_1 + m)) as usize,
	 	(255.0 * (B_1 + m)) as usize
	]
}

fn diffuse(grid: &mut Grid, boundary: &Grid, iter: usize, rate: f32) {
	for i in 0..iter {
		for y in 0..grid.size - 1 {
			for x in 0..grid.size - 1 {
				// ignore boundary
				if boundary.get(x, y) == 0.0 { continue; }
				let avg = grid.get(x-1, y) + grid.get(x+1, y) + grid.get(x, y-1) + grid.get(x, y+1);
				grid.elems[x][y] = (grid.get(x, y) + rate*avg) / (1.0 + 4.0*rate);
	}}}	
}

fn advect(grid: &mut Grid, x_vel: &Grid, y_vel: &Grid, boundary: &Grid, dt: f32) {
	
	for y in 0..grid.size-1 {
		for x in 0..grid.size-1 {
			if boundary.get(x, y) == 0.0 { continue; }
			let prev_x = (x as f32) - x_vel.get(x, y)*dt;
			let prev_y = (y as f32) - y_vel.get(x, y)*dt;

			//println!("{}, {}", prev_x, prev_y);
			let (x_cell, x_subcell) = (prev_x.floor() as usize, prev_x.fract());
			let (y_cell, y_subcell) = (prev_y.floor() as usize, prev_y.fract());

			let top = lerp(grid.get(x_cell, y_cell), grid.get(x_cell+1, y_cell), x_subcell);
			let bottom = lerp(grid.get(x_cell, y_cell+1), grid.get(x_cell+1, y_cell+1), x_subcell);

			grid.elems[x][y] = lerp(top, bottom, y_subcell);
	}}
}

fn solve_compress (x_vel: &mut Grid, y_vel: &mut Grid, div: &mut Grid, p: &mut Grid, boundary: &Grid, iter: usize) {
	let grid_size = x_vel.size;

	for y in 0..grid_size-1 {
		for x in 0..grid_size-1 {
		if boundary.get(x, y) == 0.0 { continue; }
		div.elems[x][y] = (x_vel.get(x+1, y) - x_vel.get(x-1, y) + y_vel.get(x, y+1) - y_vel.get(x, y-1)) / 2.0;
		p.elems[x][y] = 0.0;
	}}
	
	// solve for p field, a scalar field whose gradient is the divergence of the velocity field
	for i in 0..iter {
		for y in 0..grid_size-1 {
			for x in 0..grid_size-1 {
				if boundary.get(x, y) == 0.0 { continue; }					

				p.elems[x][y] = (p.get(x+1, y) + p.get(x-1, y) + p.get(x, y+1) + p.get(x, y-1) - div.get(x, y)) / 4.0;
	}}}

	// remove divergence
	for y in 0..grid_size-1 {
		for x in 0..grid_size-1 {
			if boundary.get(x, y) == 0.0 { continue; }	
			let (p_x, p_y) = ((p.get(x+1, y) - p.get(x-1, y)) / 2.0, (p.get(x, y+1) - p.get(x, y-1)) / 2.0);

			x_vel.elems[x][y] -= p_x;
			y_vel.elems[x][y] -= p_y;
	}}
}

fn render_velocities(x_grid: &Grid, y_grid: &Grid) {
	let (mut x, mut y) = (0, 0);
	let size = y_grid.size;
	while y < size {
		while x < size {
			let upper_color = vec_to_color(x_grid.get(x, y), y_grid.get(x, y));
			let lower_color = vec_to_color(x_grid.get(x, y+1), y_grid.get(x, y+1));
			
			print!(
				"\x1b[38;2;{};{};{}m\x1b[48;2;{};{};{}m▀\x1b[0m",
				upper_color[0], upper_color[1], upper_color[2],
				lower_color[0], lower_color[1], lower_color[2]
			);
			x += 1;
		}
		io::stdout().flush().unwrap();
		println!("");
		y += 2;
		x = 0;
	}
}

fn main() {
    let grid_size = 80;
    let dt = 0.02; // time step size
    let iter = 30; // numerical solver iterations
    let diff_rate = 0.0001; // diffusion factor
    let visc = 0.001; // fluid viscocity
    let frames = 2000; // number of frames to run
    
    let mut frame_counter = 0;
    let mut boundary = Grid::bounding_box(grid_size);
    let mut x_vel = Grid::new(grid_size); // x velocity in cell center
    let mut y_vel = Grid::new(grid_size); // y velocity
    let mut rho = Grid::new(grid_size); // fluid density
    let mut p = Grid::new(grid_size);
    let mut div = Grid::new(grid_size);

	// circular obstacle
    for y in 0..grid_size-1 {
    	for x in 0..grid_size-1 {
    		let w = (x as f32) - 40.0;
    		let h = (y as f32) - 40.0;
    		if (w*w + h*h).sqrt() <= 5.0 { boundary.elems[x][y] = 0.0; }
    }}
    
	// main event loop
	while frame_counter < frames {
		// constant density source
		rho.elems[10][40] = 100.0;
		
		x_vel.elems[10][40] = 1500.0;
		
		diffuse(&mut rho, &boundary, iter, diff_rate);
		diffuse(&mut x_vel, &boundary, iter, visc);
		diffuse(&mut y_vel, &boundary, iter, visc);

		solve_compress(&mut x_vel, &mut y_vel, &mut div, &mut p, &boundary, iter);

		let (x_copy, y_copy) = (x_vel.clone(), y_vel.clone()); // prevent multiple references to the same grid
		advect(&mut rho, &x_vel, &y_vel, &boundary, dt);
		advect(&mut x_vel, &x_copy, &y_vel, &boundary, dt);
		advect(&mut y_vel, &x_vel, &y_copy, &boundary, dt);
			
		solve_compress(&mut x_vel, &mut y_vel, &mut div, &mut p, &boundary, iter);

		println!("density");
		rho.render();
		println!("pressure");
		p.render();
		println!("velocity");
		render_velocities(&x_vel, &y_vel);

		frame_counter += 1;
		sleep(Duration::from_millis((20.0) as u64));
	}
}

