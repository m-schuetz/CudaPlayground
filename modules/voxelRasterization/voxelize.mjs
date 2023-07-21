
import {promises as fsp} from "fs";
const {floor} = Math;

let path = "D:/dev/pointclouds/mschuetz/lion.las";

class Point{
	constructor(){
		this.x = 0;
		this.y = 0;
		this.z = 0;
		this.r = 0;
		this.g = 0;
		this.b = 0;
	}
};

async function loadLas(path){

	let buffer = await fsp.readFile(path);

	let headerBuffer = buffer.subarray(0, 375);


	let versionMajor   = headerBuffer.readUint8(24);
	let versionMinor   = headerBuffer.readUint8(25);
	let offsetToPoints = headerBuffer.readUint32LE(96);
	let format         = headerBuffer.readUint8(104);
	let bytesPerPoint  = headerBuffer.readUint16LE(105);
	let numPoints      = headerBuffer.readUint32LE(107);

	if(versionMajor >= 1 && versionMinor >= 4){
		numPoints = Number(headerBuffer.readBigUint64LE(247));
	}
	

	let scale = {
		x: headerBuffer.readDoubleLE(131),
		y: headerBuffer.readDoubleLE(139),
		z: headerBuffer.readDoubleLE(147),
	};
	let offset = {
		x: headerBuffer.readDoubleLE(155),
		y: headerBuffer.readDoubleLE(163),
		z: headerBuffer.readDoubleLE(171),
	};
	let min = {
		x: headerBuffer.readDoubleLE(187),
		y: headerBuffer.readDoubleLE(203),
		z: headerBuffer.readDoubleLE(219),
	};
	let max = {
		x: headerBuffer.readDoubleLE(179),
		y: headerBuffer.readDoubleLE(195),
		z: headerBuffer.readDoubleLE(211),
	};

	let metadata = {
		numPoints, format, bytesPerPoint, scale, offset, min, max
	};
	
	console.log(metadata);

	let points = [];

	let realMin = {x: Infinity, y: Infinity, z: Infinity};
	let realMax = {x: -Infinity, y: -Infinity, z: -Infinity};

	let rgbOffset = 0;
	if(format === 2) rgbOffset = 20;
	if(format === 3) rgbOffset = 28;

	for(let i = 0; i < numPoints; i++){

		let X = buffer.readInt32LE(offsetToPoints + bytesPerPoint * i + 0);
		let Y = buffer.readInt32LE(offsetToPoints + bytesPerPoint * i + 4);
		let Z = buffer.readInt32LE(offsetToPoints + bytesPerPoint * i + 8);

		let x = X * scale.x + offset.x - min.x;
		let y = Y * scale.y + offset.y - min.y;
		let z = Z * scale.z + offset.z - min.z;

		realMin.x = Math.min(realMin.x, x);
		realMin.y = Math.min(realMin.y, y);
		realMin.z = Math.min(realMin.z, z);
		realMax.x = Math.max(realMax.x, x);
		realMax.y = Math.max(realMax.y, y);
		realMax.z = Math.max(realMax.z, z);

		let R = buffer.readUint16LE(offsetToPoints + bytesPerPoint * i + rgbOffset + 0);
		let G = buffer.readUint16LE(offsetToPoints + bytesPerPoint * i + rgbOffset + 2);
		let B = buffer.readUint16LE(offsetToPoints + bytesPerPoint * i + rgbOffset + 4);

		let point = new Point();
		point.x = x;
		point.y = y;
		point.z = z;
		point.r = R > 255 ? R / 256 : R;
		point.g = G > 255 ? G / 256 : G;
		point.b = B > 255 ? B / 256 : B;

		points.push(point);
	}

	metadata.realMin = realMin;
	metadata.realMax = realMax;

	return {metadata, points};
}


let {metadata, points} = await loadLas(path);

let cubicSize = Math.max(metadata.realMax.x, metadata.realMax.y, metadata.realMax.z);

// console.log(points[0]);
// console.log(metadata.realMin);
// console.log(metadata.realMax);
// console.log(cubicSize);


let gridSize = 128;
let grid = new Uint32Array(4 * gridSize ** 3);

let clamp = (value, min, max) => {
	return Math.min(Math.max(value, min), max);
};

console.log("sample voxels");
for(let point of points){

	let cx = floor(clamp(gridSize * point.x / cubicSize, 0, gridSize - 1));
	let cy = floor(clamp(gridSize * point.y / cubicSize, 0, gridSize - 1));
	let cz = floor(clamp(gridSize * point.z / cubicSize, 0, gridSize - 1));

	let voxelIndex = cx + cy * gridSize + cz * gridSize * gridSize;

	grid[4 * voxelIndex + 0] += point.r;
	grid[4 * voxelIndex + 1] += point.g;
	grid[4 * voxelIndex + 2] += point.b;
	grid[4 * voxelIndex + 3] += 1;
}

let voxels = [];

console.log("extract voxels");
for(let cx = 0; cx < gridSize; cx++)
for(let cy = 0; cy < gridSize; cy++)
for(let cz = 0; cz < gridSize; cz++)
{
	let voxelIndex = cx + cy * gridSize + cz * gridSize * gridSize;

	let R = grid[4 * voxelIndex + 0];
	let G = grid[4 * voxelIndex + 1];
	let B = grid[4 * voxelIndex + 2];
	let C = grid[4 * voxelIndex + 3];

	if(C === 0) continue;

	let r = R / C;
	let g = G / C;
	let b = B / C;

	let voxel = {
		position: {x: cx, y: cy, z: cz},
		color: {r, g, b},
	};

	voxels.push(voxel);
}

console.log(voxels[0]);

// TO CSV
let lines = [];
for(let voxel of voxels){
	let strX = voxel.position.x;
	let strY = voxel.position.y;
	let strZ = voxel.position.z;
	let strR = floor(voxel.color.r);
	let strG = floor(voxel.color.g);
	let strB = floor(voxel.color.b);
	let line = `${strX}, ${strY}, ${strZ}, ${strR}, ${strG}, ${strB}`;

	lines.push(line);
}

let csv = lines.join("\n");

await fsp.writeFile(path + "/../voxelized.csv", csv);


// TO BINARY
let binary = Buffer.alloc(voxels.length * 6);

for(let i = 0; i < voxels.length; i++){
	let voxel = voxels[i];

	binary.writeUint8(voxel.position.x, 6 * i + 0);
	binary.writeUint8(voxel.position.y, 6 * i + 1);
	binary.writeUint8(voxel.position.z, 6 * i + 2);
	binary.writeUint8(floor(voxel.color.r), 6 * i + 3);
	binary.writeUint8(floor(voxel.color.g), 6 * i + 4);
	binary.writeUint8(floor(voxel.color.b), 6 * i + 5);
}

await fsp.writeFile(path + "/../voxelized.bin", binary);
