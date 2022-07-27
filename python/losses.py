import drjit as dr
import mitsuba as mi

def l2(img, ref_img):
    return dr.mean(dr.sqr(img - ref_img))

def l1(img, ref_img):
    return dr.mean(dr.abs(img - ref_img))

def mape(img, ref_img):
    rel_error = dr.abs(img - ref_img) / dr.abs(1e-2 + dr.mean(ref_img, axis=-1))
    return dr.mean(rel_error)

def downsample(img):
    n_channels = img.shape[2]
    def linear(x, y):
        x = dr.clamp(x, 0, img.shape[0] - 1)
        y = dr.clamp(y, 0, img.shape[1] - 1)
        c_offset = dr.tile(dr.arange(mi.Int32, n_channels), img.shape[0] * img.shape[1])
        idx = y * img.shape[0] * n_channels + x * n_channels + c_offset
        return idx

    x, y = dr.meshgrid(dr.arange(mi.Int32, img.shape[0]), dr.arange(mi.Int32, img.shape[1]))
    x = dr.repeat(x, n_channels)
    y = dr.repeat(y, n_channels)
    img_linear = img.array
    r = 0.25 * (dr.gather(mi.Float, img_linear, linear(x    , y)) + \
                dr.gather(mi.Float, img_linear, linear(x + 1, y)) + \
                dr.gather(mi.Float, img_linear, linear(x    , y + 1)) + \
                dr.gather(mi.Float, img_linear, linear(x + 1, y + 1)))
    return mi.TensorXf(r, img.shape)

def multiscale(img, ref_img, loss_fn=l1, levels=4):
    loss = loss_fn(img, ref_img)
    for _ in range(levels - 1):
        img = downsample(img)
        ref_img = downsample(ref_img)
        loss += loss_fn(img, ref_img)
    return loss / levels

def multiscale_l1(img, ref_img, levels=4):
    return multiscale(img, ref_img, l1, levels)
