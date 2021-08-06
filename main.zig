const std = @import("std");
const meta = @import("std").meta;

const Atomic = std.atomic.Atomic;
const Thread = std.Thread;
const Futex = std.Thread.Futex;

const Vector = meta.Vector;
const rfloat = f64;
const Vec4 = Vector(4, rfloat);
const Color = Vector(3, u32);

const rfloat_eps = std.math.f64_epsilon;

const ZERO_VECTOR           = Vec4{ 0.0, 0.0, 0.0, 0.0 };
const ONES_VECTOR           = @splat(4, @as(rfloat, 1.0));
const RESOLUTION : usize    = 512;
const RAY_BIAS : rfloat     = 0.0005;
const SPP : usize           = 16*16; // samples per pixel
const MAX_BOUNCES : usize   = 8;
const MIN_BOUNCES : usize   = 4;
const NUM_AA : usize        = 4;
const INV_AA                = @splat(4, 1.0 / @as(rfloat, NUM_AA));

const MULTI_THREADED       = true;

const MaterialType = enum { DIFFUSE, GLOSSY, MIRROR };

const Material = struct {
    material_type   : MaterialType = MaterialType.DIFFUSE,
    diffuse         : Vec4 = ZERO_VECTOR,
    emissive        : Vec4 = ZERO_VECTOR,
    specular        : Vec4 = ZERO_VECTOR,
    exp             : rfloat = 0.0
};

fn dot(a : Vec4, b : Vec4) rfloat {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

fn cross(a : Vector(4, rfloat), b : Vector(4, rfloat)) Vector(4, rfloat) {
    return .{a[1]*b[2] - a[2]*b[1], a[2]*b[0] - a[0]*b[2], a[0]*b[1] - a[1]*b[0]};
}

fn normalize(a : Vec4) Vec4 {
    const len_sqr = dot(a, a);
    if(len_sqr > rfloat_eps) {
        const oo_len = @splat(4, 1.0 / @sqrt(len_sqr));
        return a * oo_len;
    }
    return a;
}

fn max_component(a : Vec4) rfloat {
    return std.math.max(std.math.max(a[0], a[1]), a[2]);
}

fn format_vector(ns: Vec4, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
    _ = options;
    _ = fmt;
    return std.fmt.format(writer, "[{d},{d},{d},{d}]", .{ns[0], ns[1], ns[2], ns[3]});
}
pub fn fmtVector(ns: Vec4) std.fmt.Formatter(format_vector) {
    return .{ .data = ns };
}

fn format_ray(ns: Ray, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
    _ = options;
    _ = fmt;
    return std.fmt.format(writer, "[origin={}, dir={}]", .{fmtVector(ns.origin), fmtVector(ns.dir)});
}
pub fn fmtRay(ns : Ray) std.fmt.Formatter(format_ray) {
    return .{ .data = ns };
}

const Axes = struct {
    a : Vec4,
    b : Vec4
};

// Given 1 axis, returns other 2
fn build_basis(v1 : Vec4) Axes {
    var v2 : Vec4 = undefined;

    if(@fabs(v1[0]) > @fabs(v1[1])) {
        const oo_len = 1.0 / @sqrt(v1[0]*v1[0] + v1[2]*v1[2]);
        v2 = .{-v1[2]*oo_len, 0.0, v1[0]*oo_len};
    }
    else  {
        const oo_len = 1.0 / @sqrt(v1[1]*v1[1] + v1[2]*v1[2]);
        v2 = .{0.0, v1[2]*oo_len, -v1[1]*oo_len};
    }

    return .{ .a = v2, .b = cross(v1, v2) };
}

fn transform_to_basis(vin : Vec4, vx : Vec4, vy : Vec4, vz : Vec4) Vec4 {
    const sx = @splat(4, vin[0]);
    const sy = @splat(4, vin[1]);
    const sz = @splat(4, vin[2]);

    return (vx * sx + vy * sy + vz * sz);
}

fn reflect(dir : Vec4, n : Vec4) Vec4 {
    const h = n * @splat(4, dot(dir, n) * 2.0);
    return h - dir;
}

fn get_color_component(x : rfloat) u32 {
    return @floatToInt(u32, std.math.pow(rfloat, std.math.clamp(x, 0.0, 1.0), 0.45) * 255.0 + 0.5);
}
fn get_color(v : Vec4) Color {
        return .{   get_color_component(v[0]),
                    get_color_component(v[1]),
                    get_color_component(v[2]) };
    }

const Sphere = struct {
    radius      : rfloat,
    center      : Vec4,
    material    : *const Material,

    radius_sqr  : rfloat = 0.0,

    pub fn is_light(self : Sphere) bool {
        return dot(self.material.emissive, self.material.emissive) > 0.0;
    }
    pub fn intersects(self : Sphere, ray : Ray) rfloat {
        const op = self.center - ray.origin;
        const b = dot(op, ray.dir);
        var d = b * b - dot(op, op) + self.radius_sqr;

        if(d < 0.0)
        {
            return 0.0;
        }

        d = @sqrt(d);
        var t = b - d;

        if(t > RAY_BIAS)
        {
            return t;
        }

        t = b + d;
        if(t > RAY_BIAS)
        {
            return t;
        }

        return 0.0;
    }
};
fn make_sphere(radius : rfloat, center : Vec4, material : *const Material) Sphere {
    return .{ .radius = radius, .center = center, .material = material, .radius_sqr = radius*radius };
}

const Camera = struct {
    forward     : Vector(4, rfloat),
    fov_scale   : rfloat
};

const IntersectResult = struct {
    objectIndex : ?usize = undefined,
    t : rfloat = std.math.f64_max
};

const Scene = struct {
    objects : std.ArrayList(Sphere),
    lights  : std.ArrayList(usize),
    camera  : *Camera,

    pub fn intersect(self : *Scene, ray : Ray) IntersectResult {
        var result : IntersectResult = .{};
        for(self.objects.items) |sphere, index| {
            const t = sphere.intersects(ray);

            if(t > 0.0 and t < result.t) {
                result.t = t;
                result.objectIndex = index;     
            }
        }
        return result;
    }

    pub fn collect_lights(self : *Scene) !void {
        for(self.objects.items) |obj, light_index| {
            if(obj.is_light()) {
                try self.lights.append(light_index);
            }
        }    
    }
};

const Ray = struct {
    origin : Vector(4, rfloat),
    dir    : Vector(4, rfloat),

    pub fn calc_intersection_point(self : Ray, t : rfloat) Vector(4, rfloat) {
        return self.origin + self.dir * @splat(4, t);
    }
};

const Context = struct {
    scene   : *Scene,
    samples : [SPP*2]rfloat
};

fn put16(buffer : []u8, v : u16) void {
    buffer[0] = @intCast(u8, v & 0xFF);
    buffer[1] = @intCast(u8, v >> 8) ;
}

fn write_tga_header(f : std.fs.File, width : u32, height : u32) !usize {
    var header : [18]u8 = .{0}**18;

    header[2] = 2; // 32-bit
    put16(header[12..], @intCast(u16, width));
    put16(header[14..], @intCast(u16, height));
    header[16] = 32;   // BPP
    header[17] = 0x20; // top down, non interlaced

    return f.write(header[0..]);
}

fn write_tga(pixels : []const u8, width : u32, height : u32) !void {
    const file = try std.fs.cwd().createFile(
        "file.tga",
        .{ },
    );
    defer file.close();

    _ = try write_tga_header(file, width, height);

    _ = try file.write(pixels);
}

fn new_normal(x : rfloat, y : rfloat, z : rfloat) Vec4 {
    const len_sqr = x*x + y*y + z*z;
    if(len_sqr > rfloat_eps) {
        const len = @sqrt(len_sqr);
        return Vec4{x, y, z, 0.0} / @splat(4, len);
    }
    return .{ x, y, z, 0.0 };
}

fn initialize_samples(samples : *[SPP*2]rfloat, rng : *std.rand.Random) void {
    const fspp = @as(rfloat, SPP);
    const xstrata = @sqrt(fspp);
    const ystrata = fspp / xstrata;

    var is : usize = 0;

    var ystep : rfloat = 0.0;
    while(ystep < ystrata) : (ystep += 1.0) {
        var xstep : rfloat = 0;
        while(xstep < xstrata) : (xstep += 1.0) {
            const fx = (xstep + rng.float(rfloat)) / xstrata;
            const fy = (ystep + rng.float(rfloat)) / ystrata;
            samples[is] = fx;
            samples[is + 1] = fy;
            is += 2;
        }
    }
}

fn tent_filter_v(v2 : rfloat) rfloat {
    if(v2 < 1.0) {
        return @sqrt(v2) - 1.0;
    }
    return 1.0 - @sqrt(2.0 - v2);
}

fn apply_tent_filter(samples : *[SPP*2]rfloat) void {
    var i : usize = 0;
    while(i < SPP) : (i += 1) {
        const x2 = samples[i*2+0] * 2.0;
        const y2 = samples[i*2+1] * 2.0;

        samples[i * 2 + 0] = tent_filter_v(x2);
        samples[i * 2 + 1] = tent_filter_v(y2);
    }
}

fn sample_lights(scene : *Scene, intersection : Vec4, normal : Vec4, ray_dir : Vec4, material : *const Material) Vec4 {
    var color = ZERO_VECTOR;

    for(scene.lights.items) |light_index| {
        const light = scene.objects.items[light_index];
        var l = light.center - intersection;
        const light_dist_sqr = dot(l, l);
        l = normalize(l);
        
        var d = dot(normal, l);

        var shadow_ray = Ray { .origin = intersection, .dir = l };
        var shadow_result =  scene.intersect(shadow_ray);
        if(shadow_result.objectIndex) |shadow_index| {
            if(shadow_index == light_index) {
                if(d > 0.0) {
                    const sin_alpha_max_sqr = light.radius_sqr / light_dist_sqr;
                    const cos_alpha_max = @sqrt(1.0 - sin_alpha_max_sqr);

                    const omega = 2.0 * (1.0 - cos_alpha_max);
                    d *= omega;

                    const c = material.diffuse * light.material.emissive;
                    color += c * @splat(4, d);
                }

                // Specular part
                if(material.material_type == MaterialType.GLOSSY or material.material_type == MaterialType.MIRROR) {
                    const reflected = reflect(l, normal);
                    d = -dot(reflected, ray_dir);
                    if(d > 0.0) {
                        const smul = @splat(4, std.math.pow(rfloat, d, material.exp));
                        const spec_color = material.specular * smul;
                        color += spec_color;
                    }
                }
            }
        }
    }
    return color;
}

fn sample_hemisphere_cosine(uu1 : rfloat, uu2 : rfloat) Vec4 {
    const phi = 2.0 * std.math.pi * uu1;
    const r = @sqrt(uu2);
    const s = @sin(phi);
    const c = @cos(phi);

    return .{ c * r, s * r, @sqrt(1.0 - r*r) };
}

fn sample_hemisphere_specular(uu1 : rfloat, uu2 : rfloat, exp : rfloat) Vec4 {
    const phi = 2.0 * std.math.pi * uu1;

    const cos_theta = std.math.pow(rfloat, 1.0 - uu2, 1.0 / (exp + 1.0));
    const sin_theta = @sqrt(1.0 - cos_theta*cos_theta);

    return .{ @cos(phi) * sin_theta, @sin(phi) * sin_theta, cos_theta };
}
fn interreflect_diffuse(normal : Vec4, intersection_point : Vec4, uu1 : rfloat, uu2 : rfloat) Ray {
    const v2v3 = build_basis(normal);

    const sampled_dir = sample_hemisphere_cosine(uu1, uu2);
    return .{   .origin = intersection_point, 
                .dir =  transform_to_basis(sampled_dir, v2v3.a, v2v3.b, normal) };
}

fn interreflect_specular(normal : Vec4, intersection_point : Vec4, uu1 : rfloat, uu2 : rfloat, exp : rfloat,
    ray : Ray) Ray {
    const view = -ray.dir;
    const reflected = normalize(reflect(view, normal));

    const v2v3 = build_basis(reflected);

    const sampled_dir = sample_hemisphere_specular(uu1, uu2, exp);

    return .{   .origin = intersection_point,
                .dir = transform_to_basis(sampled_dir, v2v3.a, v2v3.b, reflected) };
}

fn trace(ray : Ray, scene : *Scene, puu1 : rfloat, puu2 : rfloat, samples : [SPP*2]rfloat, rng : *std.rand.Random) Vec4 {

    var uu1 = puu1;
    var uu2 = puu2;

    var result = ZERO_VECTOR;
    var rr_scale = ONES_VECTOR;
    var direct = true;

    var bounce : usize = 0;
    var t_ray = ray;
    while(bounce < MAX_BOUNCES) : (bounce += 1) {
        const hit = scene.intersect(t_ray);
        if(hit.objectIndex) |objectIndex| {
            //std.debug.print("{} - hit {} {d}\n", .{bounce, objectIndex, hit.t});
            const obj = scene.objects.items[objectIndex];
            const material = obj.material;
            if(direct) {
                result += material.emissive * rr_scale;
            }
            var diffuse = material.diffuse;
            const max_diffuse = max_component(diffuse);
            if(bounce > MIN_BOUNCES or max_diffuse < rfloat_eps) {
                if(rng.float(rfloat) > max_diffuse) {
                    break;
                }
                diffuse /= @splat(4, max_diffuse);
            }
            
            const intersection_point = t_ray.calc_intersection_point(hit.t);
            var normal = (intersection_point - obj.center) / @splat(4, obj.radius);
            if(dot(normal, t_ray.dir) >= 0.0) {
                normal = -normal;
            }

            switch(material.material_type)
            {
                .DIFFUSE => {
                    direct = false;
                    const direct_light = rr_scale * sample_lights(scene, intersection_point, normal, t_ray.dir, material);
                    result += direct_light;
                    t_ray = interreflect_diffuse(normal, intersection_point, uu1, uu2);
                    rr_scale *= diffuse;
                },
                .GLOSSY => {
                    direct = false;
                    const direct_light = rr_scale * sample_lights(scene, intersection_point, normal, t_ray.dir, material);
                    result += direct_light;

                    // Specular/diffuse Russian roulette
                    const max_spec = max_component(material.specular);
                    const p = max_spec / (max_spec + max_diffuse);
                    const smult = 1.0 / p;

                    if(rng.float(rfloat) > p) { // diffuse
                        t_ray = interreflect_diffuse(normal, intersection_point, uu1, uu2);
                        const dscale = @splat(4, (1.0 / (1.0 - 1.0/smult)));
                        const color = diffuse * dscale;
                        rr_scale *= color;
                    }
                    else {
                        t_ray = interreflect_specular(normal, intersection_point, uu1, uu2, material.exp, t_ray);
                        const color = material.specular * @splat(4, smult);
                        rr_scale *= color;
                    }
                },
                .MIRROR => {
                    const view = -t_ray.dir;
                    const reflected = normalize(reflect(view, normal));
                    t_ray = .{ .origin = intersection_point, .dir = reflected };
                    rr_scale *= diffuse;
                }
            }

            const sample_index = rng.intRangeAtMost(usize, 0, SPP-1);
            uu1 = samples[sample_index*2];
            uu2 = samples[sample_index*2+1];         
        }   // if hit
        else {
            break;
        }
    }
    return result;
}

fn process_chunk(context : Context, buffer : []u8, offset : usize, chunk_size : usize) !void {
    const res = @as(rfloat, RESOLUTION);
    const camera = context.scene.camera;

    var cx = Vec4{camera.fov_scale, 0.0, 0.0 };
    var cy = normalize(cross(cx, camera.forward));
    cy = cy * @splat(4, camera.fov_scale);

    const ray_origin = Vec4{ 50.0, 52.0, 295.6 };

    var chunk_samples : [SPP*2]rfloat = undefined;
    var sphere_samples : [SPP*2]rfloat = undefined;

    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.os.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rng = &prng.random;    

    initialize_samples(&chunk_samples, rng);
    apply_tent_filter(&chunk_samples);

    const inv_spp = @splat(4, 1.0 / @as(rfloat, SPP));

    const start_x = offset % RESOLUTION;
    const start_y = offset / RESOLUTION;

    var y = start_y;
    var x = start_x;

    const dir_scale = @splat(4, @as(rfloat, 136.0));
    var pixel_offset = offset * 4;    
    const end_offset = pixel_offset + chunk_size * 4;

    while(pixel_offset < end_offset) : (pixel_offset += 4) {
        initialize_samples(&sphere_samples, rng);

        var cr = ZERO_VECTOR;
        var aa : usize = 0;
        while(aa < NUM_AA) : (aa += 1) {
            var pr = ZERO_VECTOR;

            const aax = @intToFloat(rfloat, (aa & 0x1));
            const aay = @intToFloat(rfloat, (aa >> 1));

            var s : usize = 0;
            while(s < SPP) : (s += 1) {
                const dx = chunk_samples[s * 2];
                const dy = chunk_samples[s * 2 + 1];

                const px = (((aax + 0.5 + dx) / 2.0) + @intToFloat(rfloat, x)) / res - 0.5;
                const py = -((((aay + 0.5 + dy) / 2.0) + @intToFloat(rfloat, y)) / res - 0.5);

                const ccx = cx * @splat(4, px);
                const ccy = cy * @splat(4, py);

                var ray_dir = normalize(ccx + ccy + camera.forward);

                var ray = Ray{ .origin = ray_origin + ray_dir * dir_scale, .dir = ray_dir};

                const uu1 = sphere_samples[s*2];
                const uu2 = sphere_samples[s*2+1];

                const r = trace(ray, context.scene, uu1, uu2, context.samples, rng);

                pr += (r * inv_spp);
            }
            cr += (pr * INV_AA);
        }
        var col = get_color(cr);

        buffer[pixel_offset + 3] = 0xFF;
        buffer[pixel_offset + 0] = @intCast(u8, col[2]);
        buffer[pixel_offset + 1] = @intCast(u8, col[1]);
        buffer[pixel_offset + 2] = @intCast(u8, col[0]);

        x = x + 1;
        if(x == RESOLUTION) {
            x = 0;
            y = y + 1;
        }
    }
}

const WorkItem = struct {
    context : *Context,
    buffer : *[]u8,
    offset : usize,
    chunk_size : usize
};

const WorkerThreadData = struct {
    //queue : WorkQueue = .{},
    queue : *std.atomic.Queue(WorkItem),
    job_counter : Atomic(u32) = Atomic(u32).init(0),
    cur_job_counter : u32 = 0,
    done : Atomic(bool) = Atomic(bool).init(false),
    done_count : *Atomic(u32),

    pub fn wait_for_job(self : *@This()) !void {
        var v : u32 = undefined;
        while(true) {
            v = self.job_counter.load(.Acquire);
            if(v != self.cur_job_counter) {
                break;
            }
            Futex.wait(&self.job_counter, self.cur_job_counter, null) catch unreachable;
        }
    }
    pub fn wake(self : *@This()) void {
        _ = self.job_counter.fetchAdd(1, .Release);
        Futex.wake(&self.job_counter, 1);
    }
    fn push_job_and_wake(self : *@This(), node : *std.atomic.Queue(WorkItem).Node) void {
        self.queue.put(node);
        self.wake();
    }
};

fn worker_thread(worker_data : *WorkerThreadData) !void {

    var work_item : WorkItem = undefined;
    while(!worker_data.done.load(.Acquire)) {
        if(!worker_data.queue.isEmpty()) {
            //worker_data.queue.pop(&work_item);
            work_item = worker_data.queue.get().?.data;

            try process_chunk(work_item.context.*, work_item.buffer.*, work_item.offset, work_item.chunk_size);

            _ = worker_data.done_count.fetchAdd(1, .Release);
            Futex.wake(worker_data.done_count, 1);
        }
        else {
            try worker_data.wait_for_job();
        }
    }
}

fn join_thread(t : std.Thread, data : *WorkerThreadData) void {
    data.done.store(true, .Release);
    data.wake();
    t.join();
}

fn wait_until_done(c : *Atomic(u32), goal_c : u32) !void {
    while(MULTI_THREADED) {
        const cv = c.load(.Acquire);
        if(cv == goal_c) {
            break;
        }
        Futex.wait(c, cv, null) catch unreachable;
    }
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    const allocator = &arena.allocator;

    const fov_scale = std.math.tan(@as(rfloat, 55.0 * std.math.pi / 180.0 * 0.5));
    var camera = Camera{ .forward = new_normal(0.0, -0.042612, -1.0), .fov_scale = fov_scale };

    const white = @splat(4, @as(rfloat, 0.99));
    const diffuse_grey = Material{ .diffuse = .{0.75, 0.75, 0.75} };
    const diffuse_red = Material{  .diffuse = .{0.95, 0.15, 0.15} };
    const diffuse_blue = Material{ .diffuse = .{0.25, 0.25, 0.7} };
    const diffuse_green = Material{ .diffuse = .{0.0, 0.55, 14.0/255.0} };
    const diffuse_black = Material{ };
    const diffuse_white = Material{ .diffuse = white };
    const glossy_white = Material{ .material_type = MaterialType.GLOSSY, .diffuse = .{0.3, 0.05, 0.05},
        .specular = @splat(4, @as(rfloat, 0.69)), .exp = 45.0 };
    const white_light = Material{ .emissive = @splat(4, @as(rfloat, 400)) };
    const mirror = Material{ .material_type = MaterialType.MIRROR, .diffuse = white };

    var scene = Scene{  .objects = try std.ArrayList(Sphere).initCapacity(allocator, 16),
                        .lights = try std.ArrayList(usize).initCapacity(allocator, 16),
                        .camera = &camera };

    try scene.objects.append(make_sphere(1e5, .{1e5 + 1.0, 40.8, 81.6}, &diffuse_red ));
    try scene.objects.append(make_sphere(1e5,.{-1e5 + 99.0, 40.8, 81.6}, &diffuse_blue ));
    try scene.objects.append(make_sphere(1e5,.{50.0, 40.8, 1e5}, &diffuse_grey ));
    try scene.objects.append(make_sphere(1e5,.{50.0, 40.8, -1e5+170.0}, &diffuse_black ));
    try scene.objects.append(make_sphere(1e5,.{50.0, 1e5, 81.6},  &diffuse_grey ));
    try scene.objects.append(make_sphere(1e5,.{50.0, -1e5 + 81.6, 81.6}, &diffuse_grey ));
    try scene.objects.append(make_sphere(16.5,.{27.0, 16.5, 57.0}, &mirror ));
    try scene.objects.append(make_sphere(10.5,.{17.0, 10.5, 97.0}, &diffuse_green ));
    try scene.objects.append(make_sphere(16.5,.{76.0, 16.5, 78.0}, &glossy_white ));
    try scene.objects.append(make_sphere(8.5, .{82.0, 8.5, 108.0}, &diffuse_white ));
    try scene.objects.append(make_sphere(1.5, .{50.0, 81.6-16.5, 81.6}, &white_light ));

    try scene.collect_lights();

    var prng = std.rand.DefaultPrng.init(blk: {
        var seed: u64 = undefined;
        try std.os.getrandom(std.mem.asBytes(&seed));
        break :blk seed;
    });
    const rng = &prng.random;    

    var context = Context { .scene = &scene, .samples = undefined };
    initialize_samples(&context.samples, rng);

    const num_pixels = RESOLUTION*RESOLUTION;

    var framebuffer = std.ArrayList(u8).init(allocator);
    try framebuffer.appendNTimes(0, num_pixels*4);

    const core_count = try Thread.getCpuCount();
    std.debug.print("Has {} cores\n", .{core_count});
    var worker_count = core_count;

    var worker_data = try std.ArrayList(WorkerThreadData).initCapacity(allocator, worker_count);
    var threads = try std.ArrayList(std.Thread).initCapacity(allocator, worker_count);
    var done_count = Atomic(u32).init(0);
    var work_queue = std.atomic.Queue(WorkItem).init();
    if(MULTI_THREADED)
    {
        var i : usize = 0;
        while(i < worker_count) : (i += 1) {
            worker_data.appendAssumeCapacity(.{ .done_count = &done_count, .queue = &work_queue });
            threads.appendAssumeCapacity(try std.Thread.spawn(.{}, worker_thread, .{&worker_data.items[i]}));
        }
    }

    const chunk_size : usize = 256;
    const num_chunks = num_pixels / chunk_size;

    const start_time = std.time.milliTimestamp();
    if(MULTI_THREADED)
    {
        var chunk_i : usize = 0;
        var thread_i : usize = 0;
        while(chunk_i < num_chunks) : (chunk_i += 1) {
            const node = allocator.create(std.atomic.Queue(WorkItem).Node) catch unreachable;
            node.* = .{
                .prev = undefined,
                .next = undefined,
                .data = .{
                    .context = &context,
                    .buffer = &framebuffer.items,
                    .offset = chunk_i * chunk_size,
                    .chunk_size = chunk_size
                }
            };
            worker_data.items[thread_i].push_job_and_wake(node);
            thread_i = (thread_i + 1) % worker_count;
        }
    }
    else {
        try process_chunk(context, framebuffer.items, 0, num_pixels);
    }
    
    //for(work_items.items) | work_item| {
        //try process_chunk(work_item.context.*, work_item.buffer.*, work_item.offset, work_item.chunk_size);
    //}

    try wait_until_done(&done_count, num_chunks);

    const time_taken = std.time.milliTimestamp() - start_time;
    std.debug.print("Took {d} seconds\n", .{@intToFloat(f32, time_taken)/1000.0});

    try write_tga(framebuffer.items, RESOLUTION, RESOLUTION);

    for(threads.items) |thread, index| {
        join_thread(thread, &worker_data.items[index]);
    }

    //const pixels = [_]u8{0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF,0xFF};
    //try write_tga(pixels[0..], 2, 2);
}
